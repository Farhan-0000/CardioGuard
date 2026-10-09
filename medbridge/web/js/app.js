/**
 * MedBridge chat SPA — application controller.
 *
 * Wires the store, API client and UI components together:
 * lazy session creation, POST-SSE streaming with a non-streaming fallback,
 * conversation switching (hash routes #/c/<id>), regenerate, feedback,
 * patient/clinician mode, theming, consent and data actions.
 *
 * Writes to one conversation (turn, regenerate, manual BP reading) never run
 * concurrently, including a stopped turn the server is still finishing
 * (writeguard.js). The browser tab title never contains message text.
 */

import { ApiError, createApi } from './api.js';
import { classifyEraseError, errorMessage, mayStillRunOnServer, rateLimitText, retryAfterSeconds } from './errors.js';
import { dateStamp, documentTitle, neutralConversationTitle, titleFromMessage } from './format.js';
import { conversationToMarkdown, normalizeMessage, normalizePublicConfig } from './normalize.js';
import { tokenText } from './sse.js';
import { createStore } from './store.js';
import { createComposer } from './ui/composer.js';
import { createContextPanel } from './ui/context.js';
import { byId, copyText, downloadText, h, prefersReducedMotion } from './ui/dom.js';
import { createSourceDrawer } from './ui/drawer.js';
import { emergencyCard } from './ui/emergency.js';
import { hydrateIcons, icon } from './ui/icons.js';
import { createAssistantMessage, findCitation, userMessage } from './ui/messages.js';
import { confirmDialog, openModal, showToast } from './ui/overlay.js';
import { createSidebar } from './ui/sidebar.js';
import { initThemeToggle } from './ui/theme.js';
import { busyMessage, countAssistantMessages, createWriteGuard, waitForSavedTurn } from './writeguard.js';

const NARROW = window.matchMedia('(max-width: 899px)');

const api = createApi();
const store = createStore();
const writeGuard = createWriteGuard();

/**
 * @typedef {import('./store.js').Conversation} Conversation
 * @typedef {ReturnType<typeof normalizeMessage>} Message
 */
const state = {
  config: normalizePublicConfig(null),
  /** @type {Conversation | null} */
  conversation: null,
  /** @type {Message[]} */
  messages: [],
  busy: false,
  /** @type {AbortController | null} */
  controller: null,
  loadSeq: 0,
  /** @type {Promise<Conversation> | null} pending session creation (shared by callers) */
  creating: null,
};

const el = {
  app: byId('app'),
  thread: byId('thread'),
  scroller: byId('chat-scroll'),
  welcome: byId('welcome'),
  title: byId('chat-title'),
  jump: byId('jump-bottom'),
  sidebarToggle: byId('sidebar-toggle'),
  contextToggle: byId('context-toggle'),
  themeToggle: byId('theme-toggle'),
  scrim: byId('scrim'),
};

// ---------------------------------------------------------------------------
// Mode & layout
// ---------------------------------------------------------------------------

/** @param {'patient' | 'clinician'} mode */
function applyMode(mode) {
  document.body.dataset.mode = mode;
  for (const button of document.querySelectorAll('[data-mode-option]')) {
    button.setAttribute('aria-checked', String(button.getAttribute('data-mode-option') === mode));
  }
  for (const panel of el.thread.querySelectorAll('details.why')) {
    /** @type {HTMLDetailsElement} */ (panel).open = mode === 'clinician';
  }
}

function sidebarOpen() {
  return el.app.dataset.sidebar === 'open';
}

/** @param {boolean} open */
function setSidebar(open) {
  el.app.dataset.sidebar = open ? 'open' : 'closed';
  el.sidebarToggle.setAttribute('aria-expanded', String(open));
  if (!NARROW.matches) store.setSidebarCollapsed(!open);
  updateScrim();
  if (open && NARROW.matches) byId('new-chat').focus();
}

function updateScrim() {
  const overlayOpen = NARROW.matches && sidebarOpen();
  const contextOverlay = window.matchMedia('(max-width: 1199px)').matches && contextPanel.isOpen;
  el.scrim.hidden = !(overlayOpen || contextOverlay);
}

// ---------------------------------------------------------------------------
// Thread rendering helpers
// ---------------------------------------------------------------------------

function nearBottom() {
  const s = el.scroller;
  return s.scrollHeight - s.scrollTop - s.clientHeight < 140;
}

/** @param {boolean} [force] */
function scrollToBottom(force = false) {
  if (!force && !nearBottom()) {
    el.jump.hidden = false;
    return;
  }
  el.scroller.scrollTo({ top: el.scroller.scrollHeight, behavior: prefersReducedMotion() || !force ? 'auto' : 'smooth' });
}

/** @param {boolean} visible */
function showWelcome(visible) {
  el.welcome.hidden = !visible;
  el.thread.hidden = visible;
}

function markLastAssistant() {
  const answers = [...el.thread.querySelectorAll('.msg--assistant')];
  answers.forEach((node, i) => node.classList.toggle('is-last', i === answers.length - 1));
}

function setTitle() {
  // The in-page header may show the (browser-local) conversation title; the tab
  // title must not, because browsers keep page titles in their history.
  el.title.textContent = state.conversation?.title ?? 'New conversation';
  document.title = documentTitle(state.conversation, state.config.appName);
}

/**
 * Why writes to conversation `id` are paused right now, or null.
 * @param {string | null | undefined} id
 */
function writeBlocked(id) {
  const kind = writeGuard.activeKind(id);
  return kind ? busyMessage(kind) : null;
}

/** Reflect the open conversation's write lock in the composer and the BP form. */
function syncWriteState() {
  const reason = writeBlocked(state.conversation?.id);
  composer.setHold(!state.busy && reason ? reason : null);
  contextPanel.syncWriteState();
}

/** @param {{streaming: boolean, onRetry?: () => void, onReload?: () => void}} options */
function newAssistant(options) {
  return createAssistantMessage(
    {
      config: state.config,
      mode: () => store.getMode(),
      onCopy: async (message) => {
        const ok = await copyText(message.content);
        showToast(ok ? 'Answer copied' : 'Copy failed — select the text instead', { tone: ok ? 'success' : 'error' });
      },
      onRegenerate: () => regenerate(),
      onFeedback: (message, rating, buttons) => feedback(message, rating, buttons),
    },
    options,
  );
}

/**
 * Inline notice in the thread (errors, unavailable conversations).
 * @param {string} text
 * @param {...(Node | null)} actions
 */
function threadNotice(text, ...actions) {
  const node = h('div', { class: 'thread-notice', role: 'status' }, icon('info', { size: 16 }), h('span', { text }), ...actions);
  el.thread.append(node);
  return node;
}

/**
 * Render a stored message (history).
 * @param {Message} message
 */
function renderStored(message) {
  if (message.role === 'user') {
    el.thread.append(userMessage(message));
    return;
  }
  const view = newAssistant({ streaming: false });
  view.finalize(message);
  el.thread.append(view.el);
}

// ---------------------------------------------------------------------------
// Sessions & conversations
// ---------------------------------------------------------------------------

/**
 * Create the server session lazily (on the first message or BP reading).
 * Concurrent callers share one creation, so two writes never make two sessions.
 * @param {string} title
 * @returns {Promise<Conversation>}
 */
async function ensureSession(title) {
  if (state.conversation) return state.conversation;
  if (!state.creating) {
    state.creating = createConversation(title).finally(() => {
      state.creating = null;
    });
  }
  return state.creating;
}

/**
 * @param {string} title
 * @returns {Promise<Conversation>}
 */
async function createConversation(title) {
  const data = await api.createSession();
  const id = String(data?.session_id ?? '');
  if (!id) throw new ApiError('Session could not be created', { code: 'NO_SESSION' });
  const now = Date.now();
  /** @type {Conversation} */
  const conversation = { id, token: data?.session_token ?? null, title, createdAt: now, updatedAt: now };
  store.saveConversation(conversation);
  state.conversation = conversation;
  history.replaceState(null, '', `#/c/${encodeURIComponent(id)}`);
  sidebar.render(id);
  setTitle();
  syncWriteState();
  return conversation;
}

/** @param {string | undefined} [id] - defaults to the open conversation */
function touchConversation(id = state.conversation?.id) {
  if (!id) return;
  const updated = store.updateConversation(id, { updatedAt: Date.now() });
  if (updated && state.conversation?.id === id) state.conversation = updated;
  sidebar.render(state.conversation?.id ?? null);
}

function abortStream() {
  state.controller?.abort();
  state.controller = null;
}

function newChat() {
  abortStream();
  state.loadSeq += 1;
  state.conversation = null;
  state.messages = [];
  el.thread.replaceChildren();
  showWelcome(true);
  setTitle();
  syncWriteState();
  if (location.hash) history.replaceState(null, '', location.pathname);
  sidebar.render(null);
  contextPanel.refresh();
  drawer.close();
  if (NARROW.matches) setSidebar(false);
  composer.focus();
}

/**
 * @param {string} id
 * @param {{force?: boolean}} [options] - reload even if it is already open
 */
async function openConversation(id, { force = false } = {}) {
  const conversation = store.getConversation(id);
  if (!conversation) {
    newChat();
    return;
  }
  if (!force && state.conversation?.id === id && state.messages.length) return;
  abortStream();
  const seq = ++state.loadSeq;
  state.conversation = conversation;
  state.messages = [];
  setTitle();
  syncWriteState();
  sidebar.render(id);
  showWelcome(false);
  drawer.close();
  if (NARROW.matches) setSidebar(false);
  el.thread.replaceChildren(h('div', { class: 'skeleton', 'aria-label': 'Loading conversation' }, h('i'), h('i'), h('i')));
  try {
    const data = await api.getHistory(conversation);
    if (seq !== state.loadSeq) return;
    state.messages = (Array.isArray(data?.messages) ? data.messages : []).map((m) => normalizeMessage(m));
    el.thread.replaceChildren();
    if (!state.messages.length) threadNotice('No messages yet in this conversation.');
    state.messages.forEach(renderStored);
    markLastAssistant();
    requestAnimationFrame(() => scrollToBottom(true));
  } catch (error) {
    if (seq !== state.loadSeq) return;
    el.thread.replaceChildren();
    const err = toApiError(error);
    if (err.sessionGone) {
      const remove = h('button', { type: 'button', class: 'btn btn--ghost btn--sm', text: 'Remove from list' });
      remove.addEventListener('click', () => removeLocal(id));
      threadNotice('This conversation is no longer available on the server — it was erased or has expired.', remove);
    } else if (err.accessDenied) {
      // The token was rejected: the server copy still exists and was NOT erased.
      const erase = h('button', { type: 'button', class: 'btn btn--ghost btn--sm', text: 'Try to delete it' });
      erase.addEventListener('click', () => eraseConversation(conversation, deleteCopy(conversation)));
      const remove = h('button', { type: 'button', class: 'btn btn--ghost btn--sm', text: 'Remove from this browser only' });
      remove.addEventListener('click', () => offerLocalRemoval(conversation));
      threadNotice(
        `Access to this conversation has expired in this browser. Its stored copy was not erased; ${retentionText()}`,
        erase,
        remove,
      );
    } else {
      threadNotice(errorMessage(err, 'Could not load this conversation.'));
    }
  }
  contextPanel.refresh();
}

/** Re-fetch the open conversation from the server (e.g. after a stopped stream). */
function reloadCurrent() {
  if (state.conversation && !state.busy) openConversation(state.conversation.id, { force: true });
}

function route() {
  const match = /^#\/c\/(.+)$/.exec(location.hash);
  if (match) openConversation(decodeURIComponent(match[1]));
  else if (state.conversation) newChat();
}

// ---------------------------------------------------------------------------
// Sending, streaming, regenerating
// ---------------------------------------------------------------------------

/** @param {boolean} value */
function setBusy(value) {
  state.busy = value;
  composer.setBusy(value);
  el.thread.setAttribute('aria-busy', String(value));
  syncWriteState();
}

/**
 * @param {unknown} error
 * @returns {ApiError}
 */
function toApiError(error) {
  return error instanceof ApiError ? error : new ApiError(String(error));
}

/**
 * Show a failed turn: the server's actionable message for refused requests
 * (409/413/422/429, the latter with a Retry-After countdown on "Try again"),
 * the pre-vetted safe fallback for outages and network failures.
 * @param {ReturnType<typeof newAssistant>} view
 * @param {ApiError} err
 */
function failTurn(view, err) {
  const retryAfter = retryAfterSeconds(err);
  view.fail(errorMessage(err), 'failed', retryAfter ? { retryAfter, retryText: rateLimitText } : {});
}

/**
 * Stream one turn into `view`. Falls back to `fallback()` when the stream
 * could not start (no events received).
 *
 * `pending` is true when the turn did not finish here but may still be
 * running on the server (stopped, or the connection dropped): the server then
 * finishes and saves it on its own.
 * @param {ReturnType<typeof newAssistant>} view
 * @param {(signal: AbortSignal, onEvent: (e: {type: string, data: any}) => void) => Promise<void>} start
 * @param {(signal: AbortSignal) => Promise<any>} fallback
 * @returns {Promise<{message: Message | null, pending: boolean}>}
 */
async function runTurn(view, start, fallback) {
  const controller = new AbortController();
  state.controller = controller;
  /** @type {Message | null} */
  let final = null;
  let gotEvent = false;
  /** @type {ApiError | null} */
  let streamError = null;
  /** @param {{type: string, data: any}} event */
  const onEvent = ({ type, data }) => {
    gotEvent = true;
    if (type === 'stage') view.stage(data ?? {});
    else if (type === 'meta') view.meta(data ?? {});
    else if (type === 'token') {
      view.token(tokenText(data));
      scrollToBottom();
    } else if (type === 'done') {
      final = normalizeMessage(data);
      view.finalize(final, { live: true });
    } else if (type === 'error') {
      // The server follows an error event with a safe `done`; remember it in case it does not.
      streamError = new ApiError(String(data?.message ?? 'Streaming failed'), {
        code: String(data?.error_code ?? 'STREAM_ERROR'),
        safeFallback: data?.safe_fallback ?? data?.message,
      });
    }
  };
  try {
    await start(controller.signal, onEvent);
    if (!final) throw streamError ?? new ApiError('The response ended unexpectedly.', { code: 'STREAM_INCOMPLETE' });
  } catch (error) {
    const err = toApiError(error);
    if (err.aborted || controller.signal.aborted) {
      view.fail('Response stopped. MedBridge may still finish and save this answer in the background.', 'stopped');
      return { message: null, pending: true };
    }
    const canFallback = !gotEvent && !err.sessionUnavailable && ![400, 409, 413, 422, 429].includes(err.status);
    if (canFallback) {
      try {
        final = normalizeMessage(await fallback(controller.signal));
        view.finalize(final, { live: true });
      } catch (fallbackError) {
        const ferr = toApiError(fallbackError);
        if (ferr.aborted || controller.signal.aborted) {
          view.fail('Response stopped. MedBridge may still finish and save this answer in the background.', 'stopped');
          return { message: null, pending: true };
        }
        failTurn(view, ferr);
        return { message: null, pending: mayStillRunOnServer(ferr) };
      }
    } else {
      failTurn(view, err);
      return { message: null, pending: mayStillRunOnServer(err) };
    }
  } finally {
    if (state.controller === controller) state.controller = null;
  }
  return { message: final, pending: false };
}

/**
 * Keep conversation writes locked until the server has saved a turn the
 * client stopped (or lost), then show the saved answer.
 * @param {Conversation} conversation
 * @param {number} baseline - assistant messages on the server before the turn
 * @param {{release: () => void, relabel: (kind: import('./writeguard.js').WriteKind) => void}} lock
 */
async function finishInBackground(conversation, baseline, lock) {
  lock.relabel('finishing');
  let outcome;
  try {
    outcome = await waitForSavedTurn({
      fetchHistory: () => api.getHistory(conversation),
      baseline,
      isCancelled: () => !store.getConversation(conversation.id),
    });
  } finally {
    lock.release();
  }
  if (outcome === 'cancelled' || outcome === 'gone') return;
  if (state.conversation?.id === conversation.id && !state.busy) {
    await openConversation(conversation.id, { force: true });
  }
  if (outcome === 'timeout') {
    showToast('The previous answer is taking longer than expected. You can continue now; reload the conversation later to see it.');
  }
}

/** @param {string} text */
async function send(text) {
  if (state.busy) return;
  const blocked = writeBlocked(state.conversation?.id);
  if (blocked) {
    composer.setText(text);
    showToast(blocked);
    return;
  }
  showWelcome(false);
  setBusy(true);
  let conversation;
  try {
    conversation = await ensureSession(titleFromMessage(text));
  } catch (error) {
    setBusy(false);
    composer.setText(text);
    showToast(errorMessage(toApiError(error), 'Could not start a conversation.'), { tone: 'error' });
    if (!state.messages.length) showWelcome(true);
    return;
  }
  const lock = writeGuard.acquire(conversation.id, 'turn');
  if (!lock) {
    setBusy(false);
    composer.setText(text);
    showToast(writeBlocked(conversation.id) ?? busyMessage(null));
    return;
  }
  const baseline = countAssistantMessages(state.messages);
  const userMsg = normalizeMessage({ role: 'user', content: text, timestamp: new Date().toISOString() });
  state.messages.push(userMsg);
  const userNode = userMessage(userMsg);
  el.thread.append(userNode);
  const view = newAssistant({
    streaming: true,
    onRetry: () => {
      if (state.busy) return;
      userNode.remove();
      view.el.remove();
      const idx = state.messages.lastIndexOf(userMsg);
      if (idx >= 0) state.messages.splice(idx, 1);
      send(text);
    },
    onReload: reloadCurrent,
  });
  el.thread.append(view.el);
  scrollToBottom(true);

  const { message: final, pending } = await runTurn(
    view,
    (signal, onEvent) => api.streamMessage(conversation, text, signal, onEvent),
    (signal) => api.sendMessage(conversation, text, signal),
  );
  const stillOpen = state.conversation?.id === conversation.id;
  if (final && stillOpen) state.messages.push(final);
  if (pending) finishInBackground(conversation, baseline, lock);
  else lock.release();
  if (stillOpen) markLastAssistant();
  touchConversation(conversation.id);
  setBusy(false);
  if (!stillOpen) return;
  scrollToBottom();
  contextPanel.refresh();
  composer.focus();
}

async function regenerate() {
  const conversation = state.conversation;
  if (state.busy || !conversation) return;
  const blocked = writeBlocked(conversation.id);
  if (blocked) {
    showToast(blocked);
    return;
  }
  const nodes = el.thread.querySelectorAll('.msg--assistant');
  const lastNode = nodes[nodes.length - 1];
  if (!lastNode) return;
  const lock = writeGuard.acquire(conversation.id, 'regenerate');
  if (!lock) return;
  const baseline = countAssistantMessages(state.messages);
  setBusy(true);
  const view = newAssistant({ streaming: true, onRetry: () => regenerate(), onReload: reloadCurrent });
  lastNode.replaceWith(view.el);
  const { message: final, pending } = await runTurn(
    view,
    (signal, onEvent) => api.streamRegenerate(conversation, signal, onEvent),
    (signal) => api.regenerate(conversation, signal),
  );
  const stillOpen = state.conversation?.id === conversation.id;
  if (final && stillOpen) {
    const idx = state.messages.map((m) => m.role).lastIndexOf('assistant');
    if (idx >= 0) state.messages[idx] = final;
    else state.messages.push(final);
  }
  if (pending) finishInBackground(conversation, baseline, lock);
  else lock.release();
  if (stillOpen) markLastAssistant();
  touchConversation(conversation.id);
  setBusy(false);
}

/**
 * Feedback with an optional comment.
 * @param {Message} message
 * @param {1 | -1} rating
 * @param {HTMLButtonElement[]} buttons
 */
async function feedback(message, rating, buttons) {
  const conversation = state.conversation;
  if (!conversation || !message.messageId) return;
  const comment = /** @type {HTMLTextAreaElement} */ (h('textarea', {
    class: 'textarea',
    rows: 4,
    maxlength: 1000,
    placeholder: rating === 1 ? 'What was helpful? (optional)' : 'What was wrong or missing? (optional)',
    'aria-label': 'Optional comment',
  }));
  const result = await openModal({
    title: rating === 1 ? 'Thanks — what worked well?' : 'Help us improve this answer',
    body: [
      h('p', { class: 'muted', text: 'Your rating is stored with this conversation and used to evaluate MedBridge. Please do not include personal details.' }),
      comment,
    ],
    actions: [
      { label: 'Cancel', variant: 'ghost', value: null },
      { label: 'Send feedback', variant: 'primary', value: 'send' },
    ],
  });
  if (result !== 'send') return;
  try {
    await api.sendFeedback(conversation, message.messageId, rating, comment.value.trim() || undefined);
    buttons.forEach((b, i) => b.setAttribute('aria-pressed', String((i === 0 ? 1 : -1) === rating)));
    showToast('Thanks — your feedback was saved', { tone: 'success' });
  } catch (error) {
    showToast(errorMessage(toApiError(error), 'Feedback could not be saved.'), { tone: 'error' });
  }
}

// ---------------------------------------------------------------------------
// Data actions
// ---------------------------------------------------------------------------

/** When stored data is purged without an explicit erase. */
function retentionText() {
  const days = state.config.retentionDays;
  return days
    ? `it is deleted automatically after ${days} days of inactivity.`
    : 'it is deleted automatically after a period of inactivity.';
}

/** @param {Conversation} conversation */
function deleteCopy(conversation) {
  return {
    title: 'Delete conversation?',
    message: `“${conversation.title}” will be erased from the server (its encryption key is destroyed) and removed from this browser. This cannot be undone.`,
    confirmLabel: 'Delete',
  };
}

/**
 * Drop a conversation (its access key and title) from this browser.
 * @param {string} id
 */
function removeLocal(id) {
  store.removeConversation(id);
  if (state.conversation?.id === id) newChat();
  else sidebar.render(state.conversation?.id ?? null);
}

/**
 * Replace the browser-local title (taken from the first, unredacted message)
 * with a neutral label. Clearing data always does this, even when the server
 * erase fails and the record (with its access key) has to stay.
 * @param {Conversation} conversation
 */
function scrubLocalTitle(conversation) {
  const updated = store.updateConversation(conversation.id, { title: neutralConversationTitle(conversation.createdAt) });
  if (updated && state.conversation?.id === conversation.id) {
    state.conversation = updated;
    setTitle();
  }
  sidebar.render(state.conversation?.id ?? null);
}

/**
 * The token was rejected (expired / foreign key): the server copy could not be
 * erased. Removing the local record is offered only as an explicit, labelled step.
 * @param {Conversation} conversation
 */
async function offerLocalRemoval(conversation) {
  const remove = await confirmDialog({
    title: 'Could not erase on the server',
    message:
      'Access to this conversation has expired in this browser, so MedBridge could not erase its stored copy. ' +
      `The copy stays encrypted on the server; ${retentionText()} ` +
      'You can remove the conversation from this browser, but that does not erase the server copy.',
    confirmLabel: 'Remove from this browser only',
    danger: true,
  });
  if (!remove) return;
  removeLocal(conversation.id);
  showToast('Removed from this browser only. The server copy was not erased.');
}

/**
 * Delete a conversation on the server (crypto-shredding) and locally.
 * Success is reported only when the server erased it (204) or no longer has
 * it (404/410). A rejected token (401/403) means nothing was erased: the
 * local record and its access key are kept and the user is told so.
 * @param {Conversation} conversation
 * @param {{title: string, message: string, confirmLabel: string}} copy
 */
async function eraseConversation(conversation, copy) {
  const ok = await confirmDialog({ ...copy, danger: true });
  if (!ok) return;
  scrubLocalTitle(conversation);
  try {
    await api.deleteSession(conversation);
  } catch (error) {
    const err = toApiError(error);
    const outcome = classifyEraseError(err);
    if (outcome === 'gone') {
      removeLocal(conversation.id);
      showToast('Removed from this browser (the server no longer had this conversation).');
    } else if (outcome === 'denied') {
      showToast('Could not erase: access to this conversation has expired.', { tone: 'error' });
      await offerLocalRemoval(conversation);
    } else {
      showToast(errorMessage(err, 'Could not erase the conversation. Please try again.'), { tone: 'error' });
    }
    return;
  }
  removeLocal(conversation.id);
  showToast('Conversation erased', { tone: 'success' });
}

function exportConversation() {
  if (!state.conversation) return;
  const md = conversationToMarkdown({
    title: state.conversation.title,
    appName: state.config.appName,
    messages: state.messages,
  });
  downloadText(`medbridge-conversation-${dateStamp()}.md`, md);
}

async function downloadSummary() {
  if (!state.conversation) return;
  try {
    const md = await api.getSummaryMarkdown(state.conversation);
    downloadText(`medbridge-clinician-summary-${dateStamp()}.md`, md);
  } catch (error) {
    showToast(errorMessage(toApiError(error), 'The summary could not be created.'), { tone: 'error' });
  }
}

/**
 * Emergency card after a manual BP entry in the severe range.
 * @param {{systolic: number, diastolic: number}} reading
 * @param {string | null} advice - pre-vetted server advice text
 */
function showCrisis(reading, advice) {
  showWelcome(false);
  const node = h('div', { class: 'msg msg--system' }, emergencyCard(state.config, {
    title: `Your reading of ${reading.systolic}/${reading.diastolic} mmHg is in the severe range`,
    detail: advice,
    lists: false,
    live: true,
  }));
  el.thread.append(node);
  scrollToBottom(true);
}

// ---------------------------------------------------------------------------
// Consent & configuration
// ---------------------------------------------------------------------------

/** @param {boolean} firstVisit */
function showPrivacy(firstVisit) {
  const cfg = state.config;
  const item = (iconName, title, text) =>
    h('li', null, h('span', { class: 'consent__icon' }, icon(iconName, { size: 18 })), h('div', null, h('strong', { text: title }), h('p', { text })));
  const body = [
    h('p', { text: `${cfg.appName} gives general, guideline-based information about blood pressure and heart health. It does not diagnose, prescribe or replace your clinician.` }),
    h('ul', { class: 'consent' },
      item('file', 'What is stored', 'Your messages and the health details you share (such as readings and medicines) are stored so the conversation can continue. Names, phone numbers, emails and ID numbers are redacted automatically first.'),
      item('lock', 'Encrypted', 'Stored conversation data is encrypted with a key that belongs to this conversation only.'),
      cfg.localOnly
        ? item('shield', 'Processed on this computer', 'AI processing runs locally — your messages are not sent to external AI services.')
        : item('shield', 'Processing', 'AI processing uses the model provider configured by your administrator.'),
      item('trash', 'Delete any time', 'Delete a conversation or use “Clear my data”: its key is destroyed and the stored content becomes unreadable.'),
      item('user', 'Kept in this browser', 'Your conversation list (titles and access keys) is saved only in this browser.'),
    ),
    h('p', { class: 'consent__emergency' }, icon('alert', { size: 16 }), `In an emergency, call ${cfg.emergencyNumber} immediately.`),
  ];
  openModal({
    title: firstVisit ? `Welcome to ${cfg.appName}` : 'Privacy & your data',
    body,
    dismissible: !firstVisit,
    className: 'modal--consent',
    actions: [{ label: firstVisit ? 'I understand' : 'Close', variant: 'primary', value: true, autofocus: true }],
  }).then(() => {
    if (firstVisit) {
      store.setConsent();
      composer.focus();
    }
  });
}

/** @param {ReturnType<typeof normalizePublicConfig>} cfg */
function applyConfig(cfg) {
  state.config = cfg;
  for (const node of document.querySelectorAll('[data-bind="appName"]')) node.textContent = cfg.appName;
  byId('welcome-tagline').textContent = cfg.tagline;
  byId('disclaimer').textContent =
    `${cfg.appName} gives general, guideline-based information — not a diagnosis or prescription. ` +
    `In an emergency call ${cfg.emergencyNumber}.`;
  byId('local-badge').hidden = !cfg.localOnly;
  const orgs = cfg.corpusOrgs.length ? cfg.corpusOrgs.join(', ') : 'AHA/ACC, ESC, NICE, WHO';
  byId('welcome-note').textContent =
    `Answers are grounded in ${cfg.corpusCount > 0 ? `${cfg.corpusCount} ` : ''}curated cardiology and ` +
    `hypertension guideline documents (e.g., ${orgs}) and always show their sources.`;
  const suggestions = byId('suggestions');
  suggestions.replaceChildren(
    ...cfg.suggestions.map((s) => {
      const card = h('button', { type: 'button', class: 'suggestion', role: 'listitem' },
        s.title ? h('span', { class: 'suggestion__title', text: s.title }) : null,
        h('span', { class: 'suggestion__prompt', text: s.prompt }));
      card.addEventListener('click', () => send(s.prompt));
      return card;
    }),
  );
  setTitle();
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

hydrateIcons(document);
initThemeToggle(el.themeToggle, store);
applyMode(store.getMode());

const composer = createComposer({ onSubmit: (text) => send(text), onStop: () => abortStream() });
const drawer = createSourceDrawer();
const sidebar = createSidebar({
  store,
  onDelete: (conversation) => eraseConversation(conversation, deleteCopy(conversation)),
  onRenamed: (conversation) => {
    if (state.conversation?.id === conversation.id) {
      state.conversation = conversation;
      setTitle();
    }
  },
});
const contextPanel = createContextPanel({
  api,
  getSession: () => state.conversation,
  ensureSession: async (title) => {
    try {
      const conversation = await ensureSession(title);
      showWelcome(false);
      return conversation;
    } catch (error) {
      showToast(errorMessage(toApiError(error), 'Could not start a conversation.'), { tone: 'error' });
      return null;
    }
  },
  writeBlocked: (sessionId) => writeBlocked(sessionId),
  beginWrite: (sessionId) => writeGuard.acquire(sessionId, 'bp'),
  onCrisis: showCrisis,
  onSummary: downloadSummary,
  onExport: exportConversation,
  onPrint: () => window.print(),
  onClear: () => {
    if (!state.conversation) return;
    eraseConversation(state.conversation, {
      title: 'Clear my data?',
      message: 'This erases this conversation and the health details recorded in it. The encryption key is destroyed, so the stored data can never be read again.',
      confirmLabel: 'Clear my data',
    });
  },
  onToast: (message, tone) => showToast(message, { tone }),
  onVisibility: (open) => {
    el.contextToggle.setAttribute('aria-expanded', String(open));
    el.app.classList.toggle('has-context', open);
    updateScrim();
  },
});

writeGuard.subscribe(() => syncWriteState());

byId('new-chat').addEventListener('click', newChat);
el.sidebarToggle.addEventListener('click', () => setSidebar(!sidebarOpen()));
byId('sidebar-close').addEventListener('click', () => setSidebar(false));
el.contextToggle.addEventListener('click', () => contextPanel.toggle());
for (const button of document.querySelectorAll('[data-mode-option]')) {
  button.addEventListener('click', () => {
    const mode = button.getAttribute('data-mode-option') === 'clinician' ? 'clinician' : 'patient';
    store.setMode(mode);
    applyMode(mode);
  });
}
byId('open-privacy').addEventListener('click', () => showPrivacy(false));
el.scrim.addEventListener('click', () => {
  if (NARROW.matches && sidebarOpen()) setSidebar(false);
  contextPanel.close();
});
el.jump.addEventListener('click', () => {
  el.jump.hidden = true;
  scrollToBottom(true);
});
el.scroller.addEventListener('scroll', () => {
  if (nearBottom()) el.jump.hidden = true;
}, { passive: true });

// Citation chips and Sources list open the source drawer.
el.thread.addEventListener('click', (event) => {
  const target = /** @type {HTMLElement} */ (event.target).closest('.cite-chip, .source-item');
  if (!target) return;
  const citation = findCitation(target, target.getAttribute('data-marker') ?? '');
  if (citation) drawer.open(citation, target);
});

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape' || document.body.classList.contains('has-modal')) return;
  if (drawer.isOpen) drawer.close();
  else if (contextPanel.isOpen && window.matchMedia('(max-width: 1199px)').matches) contextPanel.close();
  else if (NARROW.matches && sidebarOpen()) {
    setSidebar(false);
    el.sidebarToggle.focus();
  }
});

NARROW.addEventListener('change', () => setSidebar(!NARROW.matches && !store.getSidebarCollapsed()));
window.addEventListener('hashchange', route);

setSidebar(!NARROW.matches && !store.getSidebarCollapsed());
requestAnimationFrame(() => el.app.classList.remove('is-booting'));
sidebar.render(null);
showWelcome(true);

(async () => {
  try {
    applyConfig(normalizePublicConfig(await api.getPublicConfig()));
  } catch {
    applyConfig(normalizePublicConfig(null));
  }
  if (!store.hasConsent()) showPrivacy(true);
  route();
  if (store.hasConsent()) composer.focus();
})();
