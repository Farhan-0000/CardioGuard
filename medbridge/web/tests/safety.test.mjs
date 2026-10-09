// Regression tests for the review findings on the chat SPA:
//   web-ui-1  erase with a rejected token must not be reported as success
//   web-ui-2  one write at a time per conversation (incl. a stopped turn the server finishes)
//   web-ui-4  no message text in the browser tab title
//   web-ui-5  actionable server messages for 409/422/429 (+ Retry-After), safe fallback for 5xx/network
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ApiError, createApi, parseRetryAfter } from '../js/api.js';
import {
  ACCESS_EXPIRED_TEXT,
  DEFAULT_RETRY_AFTER_SECONDS,
  SESSION_GONE_TEXT,
  classifyEraseError,
  errorMessage,
  isClientError,
  mayStillRunOnServer,
  rateLimitText,
  retryAfterSeconds,
  startCountdown,
} from '../js/errors.js';
import { documentTitle, neutralConversationTitle, titleFromMessage } from '../js/format.js';
import { normalizePublicConfig } from '../js/normalize.js';
import { busyMessage, countAssistantMessages, createWriteGuard, waitForSavedTurn } from '../js/writeguard.js';

const SAFE_FALLBACK =
  "We're unable to process your question right now. For immediate health concerns, please contact your healthcare provider or call emergency services.";

/** Fake fetch answering every request with one canned response. */
function respondWith(status, body, headers = {}) {
  return async () =>
    new Response(body === null ? null : JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json', ...headers },
    });
}

async function rejection(promise) {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  assert.fail('expected the request to fail');
}

// ---------------------------------------------------------------------------
// web-ui-1: erase outcomes
// ---------------------------------------------------------------------------

test('a rejected token (401/403) on erase is "denied", never "gone"', async () => {
  for (const status of [401, 403]) {
    const api = createApi({ fetchImpl: respondWith(status, { error_code: 'FORBIDDEN', message: 'Invalid or expired session token.', safe_fallback: SAFE_FALLBACK }) });
    const err = await rejection(api.deleteSession({ id: 's1', token: 'expired' }));
    assert.ok(err instanceof ApiError);
    assert.equal(err.accessDenied, true);
    assert.equal(err.sessionGone, false);
    assert.equal(err.sessionUnavailable, true); // still "cannot use this conversation"
    assert.equal(classifyEraseError(err), 'denied');
    assert.equal(errorMessage(err), ACCESS_EXPIRED_TEXT);
  }
});

test('only 404/410 mean the server no longer has the conversation', async () => {
  for (const status of [404, 410]) {
    const api = createApi({ fetchImpl: respondWith(status, { error_code: 'GONE', message: 'gone', safe_fallback: SAFE_FALLBACK }) });
    const err = await rejection(api.deleteSession({ id: 's1', token: 't' }));
    assert.equal(err.sessionGone, true);
    assert.equal(err.accessDenied, false);
    assert.equal(classifyEraseError(err), 'gone');
    assert.equal(errorMessage(err), SESSION_GONE_TEXT);
  }
  for (const err of [new ApiError('x', { status: 500 }), new ApiError('x', { status: 429 }), new ApiError('net')]) {
    assert.equal(classifyEraseError(err), 'failed');
  }
});

test('a successful erase (204) resolves', async () => {
  const api = createApi({ fetchImpl: async () => new Response(null, { status: 204 }) });
  assert.equal(await api.deleteSession({ id: 's1', token: 't' }), null);
});

// ---------------------------------------------------------------------------
// web-ui-5: actionable messages, Retry-After
// ---------------------------------------------------------------------------

test('429 carries Retry-After and shows a wait time, not the outage fallback', async () => {
  const api = createApi({
    fetchImpl: respondWith(
      429,
      { error_code: 'RATE_LIMITED', message: 'Too many requests. Please wait 40 s and try again.', safe_fallback: SAFE_FALLBACK },
      { 'Retry-After': '40' },
    ),
  });
  const err = await rejection(api.sendMessage({ id: 's1', token: 't' }, 'hi'));
  assert.equal(err.status, 429);
  assert.equal(err.retryAfter, 40);
  assert.equal(retryAfterSeconds(err), 40);
  assert.equal(errorMessage(err), 'Too many requests. Please wait 40 s and try again.');
  assert.notEqual(errorMessage(err), SAFE_FALLBACK);
  // missing / unusable header: a default wait, still not the fallback
  const bare = new ApiError('Too many requests.', { status: 429, safeFallback: SAFE_FALLBACK });
  assert.equal(retryAfterSeconds(bare), DEFAULT_RETRY_AFTER_SECONDS);
  assert.equal(errorMessage(bare), rateLimitText(DEFAULT_RETRY_AFTER_SECONDS));
  assert.equal(rateLimitText(0), 'You can try again now.');
  assert.equal(retryAfterSeconds(new ApiError('x', { status: 500 })), 0);
});

test('Retry-After accepts delta-seconds and HTTP dates', () => {
  assert.equal(parseRetryAfter('7'), 7);
  assert.equal(parseRetryAfter(''), null);
  assert.equal(parseRetryAfter(null), null);
  assert.equal(parseRetryAfter('soon'), null);
  const now = Date.UTC(2026, 9, 9, 12, 0, 0);
  assert.equal(parseRetryAfter(new Date(now + 30_000).toUTCString(), now), 30);
  assert.equal(parseRetryAfter(new Date(now - 30_000).toUTCString(), now), 0);
});

test('409 and 422 show the server message (422 with its field details)', async () => {
  const conflict = await rejection(
    createApi({ fetchImpl: respondWith(409, { error_code: 'HTTP_409', message: 'There is no previous message to regenerate in this session.', safe_fallback: SAFE_FALLBACK }) })
      .regenerate({ id: 's1', token: 't' }),
  );
  assert.equal(errorMessage(conflict), 'There is no previous message to regenerate in this session.');

  const invalid = await rejection(
    createApi({
      fetchImpl: respondWith(422, {
        error_code: 'VALIDATION_ERROR',
        message: 'Request validation failed.',
        details: [{ loc: ['body', 'message'], msg: 'String should have at most 2000 characters', type: 'string_too_long' }],
        safe_fallback: SAFE_FALLBACK,
      }),
    }).sendMessage({ id: 's1', token: 't' }, 'x'),
  );
  assert.deepEqual(invalid.details, [{ msg: 'String should have at most 2000 characters' }]);
  assert.equal(errorMessage(invalid), 'Request validation failed: String should have at most 2000 characters.');

  const tooLarge = new ApiError('Request body too large.', { status: 413, safeFallback: SAFE_FALLBACK });
  assert.equal(errorMessage(tooLarge), 'Request body too large.');
  // a 4xx without a server message falls back to the safe text
  assert.equal(errorMessage(new ApiError('Request failed (HTTP 400)', { status: 400, safeFallback: SAFE_FALLBACK })), SAFE_FALLBACK);
});

test('5xx, network and stream failures keep the safe fallback', async () => {
  const outage = await rejection(
    createApi({ fetchImpl: respondWith(503, { error_code: 'DATABASE_UNAVAILABLE', message: 'Database service is currently unavailable.', safe_fallback: SAFE_FALLBACK }) })
      .sendMessage({ id: 's1', token: 't' }, 'x'),
  );
  assert.equal(errorMessage(outage), SAFE_FALLBACK);
  const network = await rejection(
    createApi({ fetchImpl: async () => { throw new TypeError('Failed to fetch'); } }).sendMessage({ id: 's1', token: 't' }, 'x'),
  );
  assert.equal(network.code, 'NETWORK_ERROR');
  assert.equal(errorMessage(network), network.safeFallback);
  assert.equal(isClientError(outage), false);
  assert.equal(isClientError(network), false);
  assert.equal(isClientError(new ApiError('x', { status: 409 })), true);
  assert.equal(isClientError(new ApiError('x', { status: 403 })), false);
  assert.equal(errorMessage(null, 'fallback'), 'fallback');
  assert.equal(errorMessage(new ApiError('x', { aborted: true })), 'Request cancelled.');
});

test('countdown ticks once per second down to 0 and can be cancelled', () => {
  /** @type {(() => void) | null} */
  let tick = null;
  let cleared = 0;
  const timers = {
    setIntervalImpl: (fn) => {
      tick = fn;
      return 1;
    },
    clearIntervalImpl: () => {
      cleared += 1;
    },
  };
  const seen = [];
  startCountdown(3, (remaining) => seen.push(remaining), timers);
  tick();
  tick();
  tick();
  assert.deepEqual(seen, [3, 2, 1, 0]);
  assert.equal(cleared, 1);
  const none = [];
  startCountdown(0, (r) => none.push(r), timers);
  assert.deepEqual(none, [0]);
  const cancel = startCountdown(5, () => {}, timers);
  cancel();
  assert.equal(cleared, 2);
});

// ---------------------------------------------------------------------------
// web-ui-4: browser tab title
// ---------------------------------------------------------------------------

test('the tab title never contains the (unredacted) first message', () => {
  const first = "I'm Ramesh Kumar, 9876543210, BP 190/120 since morning";
  const createdAt = new Date(2026, 9, 9, 10, 30).getTime();
  const conversation = { id: 'c1', token: 't', title: titleFromMessage(first), createdAt, updatedAt: createdAt };
  const title = documentTitle(conversation, 'MedBridge');
  assert.ok(title.startsWith('Conversation · '), title);
  assert.ok(title.endsWith(' · MedBridge'), title);
  for (const fragment of ['Ramesh', 'Kumar', '9876543210', '190/120']) assert.ok(!title.includes(fragment), title);
  assert.ok(title.includes('2026'), title);
  assert.equal(documentTitle(null, 'MedBridge'), 'MedBridge — guideline-based heart health assistant');
  assert.equal(neutralConversationTitle(undefined), 'Conversation');
  assert.equal(neutralConversationTitle(0), 'Conversation');
  assert.ok(!neutralConversationTitle(createdAt).includes('Ramesh'));
});

test('public config may publish the retention period', () => {
  assert.equal(normalizePublicConfig(null).retentionDays, null);
  assert.equal(normalizePublicConfig({ data_retention_days: 30 }).retentionDays, 30);
  assert.equal(normalizePublicConfig({ data_retention_days: 'x' }).retentionDays, null);
  assert.equal(normalizePublicConfig({ data_retention_days: -1 }).retentionDays, null);
});

// ---------------------------------------------------------------------------
// web-ui-2: one write at a time per conversation
// ---------------------------------------------------------------------------

test('the write guard allows one write per conversation and notifies changes', () => {
  const guard = createWriteGuard();
  const events = [];
  const unsubscribe = guard.subscribe((id) => events.push([id, guard.activeKind(id)]));
  const turn = guard.acquire('a', 'turn');
  assert.ok(turn);
  assert.equal(guard.isBusy('a'), true);
  assert.equal(guard.acquire('a', 'bp'), null, 'a BP entry is refused while a turn runs');
  assert.equal(guard.acquire('a', 'regenerate'), null);
  const other = guard.acquire('b', 'bp');
  assert.ok(other, 'another conversation is independent');
  turn.relabel('finishing'); // stopped: the server is still finishing the turn
  assert.equal(guard.activeKind('a'), 'finishing');
  assert.equal(guard.acquire('a', 'turn'), null, 'no new message while the stopped turn finishes');
  turn.release();
  turn.release(); // idempotent
  turn.relabel('turn'); // a stale handle cannot touch a later lock
  assert.equal(guard.isBusy('a'), false);
  const next = guard.acquire('a', 'turn');
  assert.ok(next);
  turn.release(); // stale handle does not release the new holder
  assert.equal(guard.activeKind('a'), 'turn');
  next.release();
  other.release();
  unsubscribe();
  guard.acquire('c', 'turn');
  assert.deepEqual(events, [
    ['a', 'turn'],
    ['b', 'bp'],
    ['a', 'finishing'],
    ['a', null],
    ['a', 'turn'],
    ['a', null],
    ['b', null],
  ]);
  assert.equal(guard.isBusy(null), false);
  assert.equal(guard.acquire('', 'turn'), null);
});

test('busy messages explain why sending or saving is paused', () => {
  assert.match(busyMessage('finishing'), /still finishing the previous answer/);
  assert.match(busyMessage('bp'), /reading is still being saved/);
  assert.match(busyMessage('turn'), /current answer has finished/);
  assert.match(busyMessage(null), /current answer has finished/);
});

/** Deterministic clock + sleep for waitForSavedTurn. */
function fakeClock() {
  let t = 0;
  const sleeps = [];
  return {
    now: () => t,
    sleep: async (ms) => {
      sleeps.push(ms);
      t += ms;
    },
    sleeps,
  };
}

test('a stopped turn stays locked until the server has saved it', async () => {
  const clock = fakeClock();
  const histories = [
    { messages: [{ role: 'user' }, { role: 'assistant' }, { role: 'user' }] },
    { messages: [{ role: 'user' }, { role: 'assistant' }, { role: 'user' }] },
    { messages: [{ role: 'user' }, { role: 'assistant' }, { role: 'user' }, { role: 'assistant' }] },
  ];
  let calls = 0;
  const outcome = await waitForSavedTurn({
    fetchHistory: async () => histories[Math.min(calls++, histories.length - 1)],
    baseline: 1,
    intervalMs: 1000,
    maxIntervalMs: 2000,
    sleep: clock.sleep,
    now: clock.now,
  });
  assert.equal(outcome, 'saved');
  assert.equal(calls, 3);
  assert.deepEqual(clock.sleeps, [1000, 1500]);
  assert.equal(countAssistantMessages(histories[2].messages), 2);
  assert.equal(countAssistantMessages(null), 0);
});

test('waiting ends on erase, cancellation or the time limit (errors are retried)', async () => {
  const gone = await waitForSavedTurn({
    fetchHistory: async () => {
      throw new ApiError('gone', { status: 410 });
    },
    baseline: 0,
    ...fakeClock(),
  });
  assert.equal(gone, 'gone');

  let n = 0;
  const cancelled = await waitForSavedTurn({
    fetchHistory: async () => ({ messages: [] }),
    baseline: 0,
    isCancelled: () => ++n > 2,
    ...fakeClock(),
  });
  assert.equal(cancelled, 'cancelled');

  const clock = fakeClock();
  let attempts = 0;
  const timeout = await waitForSavedTurn({
    fetchHistory: async () => {
      attempts += 1;
      if (attempts % 2) throw new ApiError('Unable to reach the MedBridge service.', { code: 'NETWORK_ERROR' });
      return { messages: [{ role: 'assistant' }] };
    },
    baseline: 1,
    intervalMs: 1000,
    maxIntervalMs: 4000,
    timeoutMs: 10_000,
    sleep: clock.sleep,
    now: clock.now,
  });
  assert.equal(timeout, 'timeout');
  assert.ok(attempts >= 3, `kept polling through errors (${attempts})`);
  assert.ok(clock.now() <= 10_000);
});

test('stopped or cut-off turns may still be saved by the server; answered errors may not', () => {
  assert.equal(mayStillRunOnServer(new ApiError('Request cancelled', { aborted: true, code: 'ABORTED' })), true);
  assert.equal(mayStillRunOnServer(new ApiError('lost', { code: 'NETWORK_ERROR' })), true);
  assert.equal(mayStillRunOnServer(new ApiError('ended', { code: 'STREAM_INCOMPLETE' })), true);
  assert.equal(mayStillRunOnServer(new ApiError('server error event', { code: 'STREAM_ERROR' })), false);
  assert.equal(mayStillRunOnServer(new ApiError('limited', { status: 429 })), false);
  assert.equal(mayStillRunOnServer(new ApiError('down', { status: 503 })), false);
  assert.equal(mayStillRunOnServer(null), false);
});
