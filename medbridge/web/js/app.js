/**
 * MedBridge chat SPA — application controller.
 *
 * Wires the store, API client and UI components together:
 * lazy session creation, POST-SSE streaming with a non-streaming fallback,
 * conversation switching (hash routes #/c/<id>), regenerate, feedback,
 * patient/clinician mode, theming, consent and data actions.
 */

import { ApiError, createApi } from './api.js';
import { dateStamp, titleFromMessage } from './format.js';
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

const NARROW = window.matchMedia('(max-width: 899px)');

const api = createApi();
const store = createStore();

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
  el.title.textContent = state.conversation?.title ?? 'New conversation';
  document.title = state.conversation
    ? `${state.conversation.title} · ${state.config.appName}`
    : `${state.config.appName} — guideline-based heart health assistant`;
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
 * @param {Node | null} [action]
 */
function threadNotice(text, action = null) {
  const node = h('div', { class: 'thread-notice', role: 'status' }, icon('info', { size: 16 }), h('span', { text }), action);
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
 * Create the server session lazily (on the first message).
 * @param {string} title
 * @returns {Promise<Conversation | null>}
 */
async function ensureSession(title) {
  if (state.conversation) return state.conversation;
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
  return conversation;
}

function touchConversation() {
  if (!state.conversation) return;
  const updated = store.updateConversation(state.conversation.id, { updatedAt: Date.now() });
  if (updated) state.conversation = updated;
  sidebar.render(state.conversation.id);
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
    const err = /** @type {ApiError} */ (error);
    if (err.sessionUnavailable) {
      const remove = h('button', { type: 'button', class: 'btn btn--ghost btn--sm', text: 'Remove from list' });
      remove.addEventListener('click', () => {
        store.removeConversation(id);
        newChat();
      });
      threadNotice('This conversation is no longer available — it may have expired or been erased.', remove);
    } else {
      threadNotice(err.safeFallback || 'Could not load this conversation.');
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
}

/**
 * Stream one turn into `view`. Falls back to `fallback()` when the stream
 * could not start (no events received).
 * @param {ReturnType<typeof newAssistant>} view
 * @param {(signal: AbortSignal, onEvent: (e: {type: string, data: any}) => void) => Promise<void>} start
 * @param {(signal: AbortSignal) => Promise<any>} fallback
 * @returns {Promise<Message | null>}
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
    const err = error instanceof ApiError ? error : new ApiError(String(error));
    if (err.aborted || controller.signal.aborted) {
      view.fail('Response stopped. MedBridge may still finish and save this answer in the background.', 'stopped');
      return null;
    }
    const canFallback = !gotEvent && !err.sessionUnavailable && ![409, 413, 422, 429].includes(err.status);
    if (canFallback) {
      try {
        final = normalizeMessage(await fallback(controller.signal));
        view.finalize(final, { live: true });
      } catch (fallbackError) {
        const ferr = /** @type {ApiError} */ (fallbackError);
        view.fail(ferr.aborted ? 'Response stopped.' : ferr.safeFallback, ferr.aborted ? 'stopped' : 'failed');
        return null;
      }
    } else {
      view.fail(
        err.sessionUnavailable
          ? 'This conversation is no longer available. Please start a new chat.'
          : err.safeFallback,
        'failed',
      );
      return null;
    }
  } finally {
    if (state.controller === controller) state.controller = null;
  }
  return final;
}

/** @param {string} text */
async function send(text) {
  if (state.busy) return;
  showWelcome(false);
  setBusy(true);
  let conversation;
  try {
    conversation = await ensureSession(titleFromMessage(text));
  } catch (error) {
    setBusy(false);
    composer.setText(text);
    showToast(/** @type {ApiError} */ (error).safeFallback || 'Could not start a conversation.', { tone: 'error' });
    if (!state.messages.length) showWelcome(true);
    return;
  }
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

  const final = await runTurn(
    view,
    (signal, onEvent) => api.streamMessage(conversation, text, signal, onEvent),
    (signal) => api.sendMessage(conversation, text, signal),
  );
  if (final) state.messages.push(final);
  markLastAssistant();
  touchConversation();
  setBusy(false);
  scrollToBottom();
  contextPanel.refresh();
  composer.focus();
}

async function regenerate() {
  const conversation = state.conversation;
  if (state.busy || !conversation) return;
  const nodes = el.thread.querySelectorAll('.msg--assistant');
  const lastNode = nodes[nodes.length - 1];
  if (!lastNode) return;
  setBusy(true);
  const view = newAssistant({ streaming: true, onRetry: () => regenerate(), onReload: reloadCurrent });
  lastNode.replaceWith(view.el);
  const final = await runTurn(
    view,
    (signal, onEvent) => api.streamRegenerate(conversation, signal, onEvent),
    (signal) => api.regenerate(conversation, signal),
  );
  if (final) {
    const idx = state.messages.map((m) => m.role).lastIndexOf('assistant');
    if (idx >= 0) state.messages[idx] = final;
    else state.messages.push(final);
  }
  markLastAssistant();
  touchConversation();
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
    showToast(/** @type {ApiError} */ (error).safeFallback || 'Feedback could not be saved.', { tone: 'error' });
  }
}

// ---------------------------------------------------------------------------
// Data actions
// ---------------------------------------------------------------------------

/**
 * Delete a conversation on the server (crypto-shredding) and locally.
 * @param {Conversation} conversation
 * @param {{title: string, message: string, confirmLabel: string}} copy
 */
async function eraseConversation(conversation, copy) {
  const ok = await confirmDialog({ ...copy, danger: true });
  if (!ok) return;
  try {
    await api.deleteSession(conversation);
    showToast('Conversation erased', { tone: 'success' });
  } catch (error) {
    const err = /** @type {ApiError} */ (error);
    if (!err.sessionUnavailable) {
      showToast(err.safeFallback || 'Could not erase the conversation. Please try again.', { tone: 'error' });
      return;
    }
    showToast('Removed from this browser (the server no longer had this conversation).');
  }
  store.removeConversation(conversation.id);
  if (state.conversation?.id === conversation.id) newChat();
  else sidebar.render(state.conversation?.id ?? null);
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
    showToast(/** @type {ApiError} */ (error).safeFallback || 'The summary could not be created.', { tone: 'error' });
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
  onDelete: (conversation) =>
    eraseConversation(conversation, {
      title: 'Delete conversation?',
      message: `“${conversation.title}” will be erased from the server (its encryption key is destroyed) and removed from this browser. This cannot be undone.`,
      confirmLabel: 'Delete',
    }),
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
      showToast(/** @type {ApiError} */ (error).safeFallback || 'Could not start a conversation.', { tone: 'error' });
      return null;
    }
  },
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
