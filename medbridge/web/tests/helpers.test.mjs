// Unit tests for formatting, BP, action and payload-normalisation helpers.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ACTION_META, ACTIONS, normalizeAction, stageShortName } from '../js/actions.js';
import { bpCategory, normalizeCategory, validateBpInput } from '../js/bp.js';
import { formatCount, formatMs, formatPercent, groupConversations, stripMarkdown, titleFromMessage } from '../js/format.js';
import { createApi, ApiError } from '../js/api.js';
import {
  FALLBACK_SUGGESTIONS,
  conversationToMarkdown,
  normalizeAudit,
  normalizeContext,
  normalizeMessage,
  normalizeMetrics,
  normalizePublicConfig,
} from '../js/normalize.js';

test('titleFromMessage truncates to 48 characters', () => {
  assert.equal(titleFromMessage('  short   question '), 'short question');
  const long = 'a'.repeat(60);
  assert.equal(titleFromMessage(long), `${'a'.repeat(48)}…`);
  assert.equal(titleFromMessage(''), 'New conversation');
});

test('groupConversations buckets Today / Previous 7 days / Older', () => {
  const now = new Date(2026, 9, 9, 15, 0).getTime();
  const day = 24 * 3600 * 1000;
  const groups = groupConversations(
    [
      { id: 'old', updatedAt: now - 30 * day },
      { id: 'today', updatedAt: now - 3600 * 1000 },
      { id: 'week', updatedAt: now - 3 * day },
    ],
    now,
  );
  assert.deepEqual(groups.map((g) => g.label), ['Today', 'Previous 7 days', 'Older']);
  assert.deepEqual(groups.map((g) => g.items[0].id), ['today', 'week', 'old']);
  assert.deepEqual(groupConversations([], now), []);
});

test('stripMarkdown flattens previews', () => {
  assert.equal(stripMarkdown('## Title\nA **bold** `code` reply [1]'), 'Title A bold code reply [1]');
  assert.equal(stripMarkdown(null), '');
});

test('formatPercent and formatCount', () => {
  assert.equal(formatPercent(0.756), '76%');
  assert.equal(formatPercent(null), '—');
  assert.equal(formatCount(1234), (1234).toLocaleString());
});

test('formatMs', () => {
  assert.equal(formatMs(5.4), '5 ms');
  assert.equal(formatMs(2450), '2.5 s');
  assert.equal(formatMs(25_000), '25 s');
  assert.equal(formatMs(null), '—');
});

test('ACC/AHA categories', () => {
  assert.equal(bpCategory(115, 75), 'normal');
  assert.equal(bpCategory(125, 75), 'elevated');
  assert.equal(bpCategory(125, 82), 'stage1');
  assert.equal(bpCategory(135, 70), 'stage1');
  assert.equal(bpCategory(145, 70), 'stage2');
  assert.equal(bpCategory(120, 95), 'stage2');
  assert.equal(bpCategory(179, 100), 'stage2');
  assert.equal(bpCategory(180, 100), 'crisis');
  assert.equal(bpCategory(150, 120), 'crisis');
  assert.equal(normalizeCategory('Stage 1 hypertension'), 'stage1');
  assert.equal(normalizeCategory('STAGE_2'), 'stage2');
  assert.equal(normalizeCategory('hypertensive_crisis'), 'crisis');
  assert.equal(normalizeCategory('Elevated'), 'elevated');
  assert.equal(normalizeCategory('normal'), 'normal');
  assert.equal(normalizeCategory(''), null);
});

test('validateBpInput mirrors server rules', () => {
  const now = Date.parse('2026-10-09T10:00:00Z');
  const ok = validateBpInput({ systolic: '142', diastolic: '91', pulse: '', measuredAt: '' }, now);
  assert.equal(ok.ok, true);
  assert.deepEqual(ok.value, { systolic: 142, diastolic: 91, pulse: null, measured_at: '2026-10-09T10:00:00.000Z' });
  const bad = validateBpInput({ systolic: '90', diastolic: '120', pulse: '400', measuredAt: '2030-01-01T00:00' }, now);
  assert.equal(bad.ok, false);
  assert.ok(bad.errors.diastolic);
  assert.ok(bad.errors.pulse);
  assert.ok(bad.errors.measuredAt);
  const missing = validateBpInput({ systolic: '', diastolic: 'abc' }, now);
  assert.ok(missing.errors.systolic);
  assert.ok(missing.errors.diastolic);
});

test('action metadata covers the five actions', () => {
  assert.deepEqual([...ACTIONS], ['ANSWER', 'GENERALIZE', 'SOFT-ASK', 'ABSTAIN', 'ESCALATE']);
  for (const a of ACTIONS) assert.ok(ACTION_META[a].tooltip.length > 10);
  assert.equal(normalizeAction('soft_ask'), 'SOFT-ASK');
  assert.equal(normalizeAction('nonsense'), null);
  assert.equal(stageShortName('context_gate'), 'Gate 1 · Context');
  assert.equal(stageShortName('new_stage'), 'new stage');
});

test('normalizePublicConfig applies fallbacks and rejects unsafe phone numbers', () => {
  const empty = normalizePublicConfig(null);
  assert.equal(empty.appName, 'MedBridge');
  assert.equal(empty.emergencyNumber, '112');
  assert.equal(empty.suggestions.length, FALLBACK_SUGGESTIONS.length);
  const cfg = normalizePublicConfig({
    app_name: 'CardioGuard',
    emergency_number: '911',
    ambulance_number: 'javascript:alert(1)',
    emergency_region_label: 'United States',
    local_only_mode: true,
    suggestions: ['What is stage 1?', { title: 'Diet', prompt: 'Salt?' }, { bad: 1 }],
  });
  assert.equal(cfg.appName, 'CardioGuard');
  assert.equal(cfg.emergencyNumber, '911');
  assert.equal(cfg.ambulanceNumber, '108');
  assert.equal(cfg.region, 'United States');
  assert.equal(cfg.localOnly, true);
  assert.deepEqual(cfg.suggestions, [
    { title: '', prompt: 'What is stage 1?' },
    { title: 'Diet', prompt: 'Salt?' },
  ]);
});

test('normalizeContext reads bp_series or snapshot readings', () => {
  const ctx = normalizeContext({
    snapshot: {
      current_medications: [{ drug: 'amlodipine', dosage: '5 mg', frequency: 'daily' }, 'lisinopril'],
      symptoms: ['headache'],
      lab_results: [{ lab_type: 'eGFR', value: 60, unit: 'mL/min' }],
      demographics: { age: 54, sex: 'female' },
    },
    bp_series: [
      { systolic: 150, diastolic: 95, measured_at: '2026-10-09T08:00:00Z', category: 'Stage 2 hypertension' },
      { systolic: 138, diastolic: 85, measured_at: '2026-10-08T08:00:00Z' },
    ],
  });
  assert.equal(ctx.series.length, 2);
  assert.equal(ctx.series[0].systolic, 138, 'sorted by time');
  assert.equal(ctx.series[0].category, 'stage1', 'computed fallback');
  assert.equal(ctx.latest.systolic, 150);
  assert.equal(ctx.latest.category, 'stage2');
  assert.deepEqual(ctx.medications, [
    { name: 'amlodipine', detail: '5 mg · daily' },
    { name: 'lisinopril', detail: '' },
  ]);
  assert.deepEqual(ctx.labs, [{ name: 'eGFR', value: '60 mL/min' }]);
  assert.equal(ctx.isEmpty, false);
  const fromSnapshot = normalizeContext({ recent_bp_readings: [{ systolic: 120, diastolic: 70 }] });
  assert.equal(fromSnapshot.series[0].category, 'elevated');
  assert.equal(normalizeContext({}).isEmpty, true);
});

test('normalizeMessage maps MessageResponse and history rows', () => {
  const m = normalizeMessage({ response_text: 'Hi [1]', action: 'ANSWER', message_id: 'abc', redaction_count: 2, citations: [{ marker: '[1]' }, 'junk'] });
  assert.equal(m.role, 'assistant');
  assert.equal(m.content, 'Hi [1]');
  assert.equal(m.citations.length, 1);
  assert.equal(m.redactionCount, 2);
  assert.equal(normalizeMessage({ role: 'user', content: 'x' }).role, 'user');
});

test('normalizeMetrics accepts nested and flat shapes', () => {
  const flat = normalizeMetrics({
    total_turns: 10,
    latency_p50_ms: 1200,
    latency_p95_ms: 4000,
    action_counts: { ANSWER: 6, ABSTAIN: 4 },
    gate1_counts: { PROCEED: 8, 'SOFT-ASK': 2 },
    gate2_counts: { ANSWER: 6 },
    latency_by_action: { ANSWER: { p50: 1000, p95: 3000, n: 6 } },
    latency_by_stage: { retrieval: { p50_ms: 80, p95_ms: 150 } },
    feedback: { up: 3, down: 1 },
  });
  assert.equal(flat.totalTurns, 10);
  assert.equal(flat.p95, 4000);
  assert.deepEqual(flat.actions, [['ANSWER', 6], ['ABSTAIN', 4]]);
  assert.equal(flat.byStage[0].p95, 150);
  assert.equal(flat.feedback.rate, 0.75);
  const nested = normalizeMetrics({
    latency: { p50: 5, p95: 9, by_action: [{ action: 'ESCALATE', p50: 2, p95: 3 }] },
    actions: { ESCALATE: { count: 2 } },
    gates: { gate1: { PROCEED: 1 } },
  });
  assert.equal(nested.totalTurns, 2);
  assert.equal(nested.p50, 5);
  assert.equal(nested.byAction[0].key, 'ESCALATE');
  assert.deepEqual(nested.gate1, [['PROCEED', 1]]);
  assert.equal(nested.feedback.rate, null);
});

test('backend shapes: MetricsResponse, PublicConfig and ContextResponse', () => {
  const metrics = normalizeMetrics({
    turns: 12,
    window: 500,
    final_actions: { ANSWER: 7, 'SOFT-ASK': 5 },
    gate_1: { PROCEED: 7, 'SOFT-ASK': 5 },
    gate_2: { ANSWER: 7 },
    gate_sources: { gate1: { llm: 11, fallback: 1 }, gate2: { llm: 7 } },
    latency: { count: 12, p50_ms: 900.5, p95_ms: 3100, mean_ms: 1200, max_ms: 4000 },
    latency_by_action: { ANSWER: { count: 7, p50_ms: 1000, p95_ms: 3000 } },
    stage_latency: { retrieval: { count: 7, p50_ms: 80, p95_ms: 120 } },
    routing: { loop_breaker: 1, fastpath: 0, small_talk: 2 },
    speculative_retrieval: { launched: 7, discarded: 2, saved_ms_p50: null },
    feedback: { count: 4, thumbs_up: 3, thumbs_down: 1, positive_rate: 0.75 },
  });
  assert.equal(metrics.totalTurns, 12);
  assert.equal(metrics.p50, 900.5);
  assert.deepEqual(metrics.actions, [['ANSWER', 7], ['SOFT-ASK', 5]]);
  assert.deepEqual(metrics.gate1Sources, [['llm', 11], ['fallback', 1]]);
  assert.equal(metrics.byAction[0].n, 7);
  assert.equal(metrics.byStage[0].key, 'retrieval');
  assert.deepEqual(metrics.feedback, { up: 3, down: 1, total: 4, rate: 0.75 });
  assert.match(metrics.windowLabel, /500/);
  assert.deepEqual(metrics.safety.map((g) => g.title), ['Routing', 'Speculative retrieval']);
  assert.deepEqual(metrics.safety[1].entries, [['launched', 7], ['discarded', 2]]);

  const cfg = normalizePublicConfig({
    app_name: 'MedBridge',
    suggested_prompts: ['Is 148/92 high?', 'What helps lower BP?'],
    corpus: { documents: 3, items: [{ title: 'A', org: 'AHA/ACC' }, { title: 'B', org: 'ESC' }, { title: 'C', org: 'AHA/ACC' }] },
  });
  assert.deepEqual(cfg.suggestions.map((s) => s.prompt), ['Is 148/92 high?', 'What helps lower BP?']);
  assert.equal(cfg.corpusCount, 3);
  assert.deepEqual(cfg.corpusOrgs, ['AHA/ACC', 'ESC']);

  const ctx = normalizeContext({
    snapshot: { current_medications: [] },
    bp_readings: [
      { systolic: 182, diastolic: 100, pulse: null, recorded_at: '2026-10-09T08:00:00+00:00', source: 'manual', category: 'crisis', category_label: 'Severe / hypertensive crisis range' },
    ],
    counts: { BP_READING: 1 },
  });
  assert.equal(ctx.latest.category, 'crisis');
  assert.equal(ctx.latest.time.toISOString(), '2026-10-09T08:00:00.000Z');
});

test('normalizeAudit accepts arrays or wrapped lists', () => {
  const rows = normalizeAudit({ items: [{ created_at: '2026-10-09T10:00:00Z', final_action: 'ANSWER', gate_1_action: 'PROCEED', latency_ms: 900, request_preview: 'My BP [NAME]' }] });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].action, 'ANSWER');
  assert.equal(rows[0].gate1, 'PROCEED');
  assert.equal(rows[0].latency, 900);
  assert.equal(normalizeAudit([{ action: 'ABSTAIN' }])[0].action, 'ABSTAIN');
  assert.deepEqual(normalizeAudit(null), []);
});

test('conversationToMarkdown exports messages and sources', () => {
  const md = conversationToMarkdown(
    {
      title: 'BP question',
      appName: 'MedBridge',
      messages: [
        normalizeMessage({ role: 'user', content: 'Is 145/92 high?' }),
        normalizeMessage({ role: 'assistant', content: 'Yes [1].', action: 'ANSWER', citations: [{ marker: '[1]', source: 'AHA/ACC 2025', section: 'Table 3', page: 12 }] }),
      ],
    },
    new Date(2026, 9, 9),
  );
  assert.match(md, /^# BP question\n/);
  assert.match(md, /\*\*You\*\*/);
  assert.match(md, /\*\*MedBridge\*\* — ANSWER/);
  assert.match(md, /- \[1\] AHA\/ACC 2025 — Table 3, p\. 12/);
});

function fakeResponse({ status = 200, body = null, type = 'application/json', stream = null }) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (k) => (k.toLowerCase() === 'content-type' ? type : null) },
    json: async () => {
      if (body === null) throw new Error('no body');
      return body;
    },
    text: async () => String(body ?? ''),
    body: stream,
  };
}

test('api client sends bearer tokens and parses ErrorResponse', async () => {
  const calls = [];
  const api = createApi({
    fetchImpl: async (url, init) => {
      calls.push({ url, init });
      if (url.endsWith('/history')) {
        return fakeResponse({ status: 401, body: { error_code: 'UNAUTHORIZED', message: 'bad token', safe_fallback: 'Start again.' } });
      }
      return fakeResponse({ body: { ok: true } });
    },
  });
  await api.sendMessage({ id: 's 1', token: 'tok' }, 'hello');
  assert.equal(calls[0].url, '/api/sessions/s%201/messages');
  assert.equal(calls[0].init.headers.Authorization, 'Bearer tok');
  assert.equal(calls[0].init.body, JSON.stringify({ message: 'hello' }));
  await assert.rejects(api.getHistory({ id: 'x', token: 'tok' }), (err) => {
    assert.ok(err instanceof ApiError);
    assert.equal(err.status, 401);
    assert.equal(err.code, 'UNAUTHORIZED');
    assert.equal(err.safeFallback, 'Start again.');
    assert.equal(err.sessionUnavailable, true);
    return true;
  });
});

test('api stream falls back to JSON body delivered as done', async () => {
  const api = createApi({ fetchImpl: async () => fakeResponse({ body: { action: 'ANSWER' } }) });
  const events = [];
  await api.streamMessage({ id: 'a', token: null }, 'q', undefined, (e) => events.push(e));
  assert.deepEqual(events, [{ type: 'done', data: { action: 'ANSWER' } }]);
});

test('api stream parses SSE bodies and maps network errors', async () => {
  const bytes = new TextEncoder().encode('event: stage\ndata: {"stage":"emergency"}\n\nevent: done\ndata: {"action":"ESCALATE"}\n\n');
  const stream = new ReadableStream({ start(c) { c.enqueue(bytes); c.close(); } });
  const api = createApi({ fetchImpl: async () => fakeResponse({ type: 'text/event-stream', stream }) });
  const events = [];
  await api.streamMessage({ id: 'a', token: 't' }, 'q', undefined, (e) => events.push(e));
  assert.deepEqual(events.map((e) => e.type), ['stage', 'done']);

  const failing = createApi({ fetchImpl: async () => { throw new TypeError('Failed to fetch'); } });
  await assert.rejects(failing.createSession(), (err) => err.code === 'NETWORK_ERROR');
  const aborted = createApi({ fetchImpl: async () => { const e = new Error('x'); e.name = 'AbortError'; throw e; } });
  await assert.rejects(aborted.createSession(), (err) => err.aborted === true);
});

test('summary markdown accepts text or JSON bodies', async () => {
  const text = createApi({ fetchImpl: async () => fakeResponse({ type: 'text/markdown', body: '# Summary' }) });
  assert.equal(await text.getSummaryMarkdown({ id: 'a', token: 't' }), '# Summary');
  const json = createApi({ fetchImpl: async () => fakeResponse({ body: { markdown: '# S2' } }) });
  assert.equal(await json.getSummaryMarkdown({ id: 'a', token: 't' }), '# S2');
});
