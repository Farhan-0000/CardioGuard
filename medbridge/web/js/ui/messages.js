/**
 * Chat message components: user bubbles and assistant answers with action
 * badge, live stepper, streaming Markdown, emergency card, sources list,
 * "Why this answer?" panel and message actions.
 */

import { ACTION_META } from '../actions.js';
import { startCountdown } from '../errors.js';
import { renderMarkdown } from '../markdown.js';
import { emergencyCard } from './emergency.js';
import { explanationPanel } from './explanation.js';
import { h } from './dom.js';
import { icon } from './icons.js';
import { createStepper } from './stepper.js';

/**
 * @typedef {ReturnType<typeof import('../normalize.js').normalizeMessage>} Message
 * @typedef {{emergencyNumber: string, ambulanceNumber: string, region: string, appName: string}} UiConfig
 * @typedef {{
 *   config: UiConfig,
 *   mode: () => 'patient' | 'clinician',
 *   onCopy: (message: Message) => void,
 *   onRegenerate: () => void,
 *   onFeedback: (message: Message, rating: 1 | -1, buttons: HTMLButtonElement[]) => void,
 * }} AssistantDeps
 */

/** Citations of each rendered assistant message, looked up on chip click. */
const citationsByMessage = new WeakMap();

/**
 * Citation for marker `n` inside the message that contains `node`.
 * @param {Element} node
 * @param {string} marker - "1", "2", ...
 * @returns {any | null}
 */
export function findCitation(node, marker) {
  const article = node.closest('.msg--assistant');
  const list = article ? citationsByMessage.get(article) : null;
  if (!list) return null;
  return list.find((c) => String(c.marker ?? '').replace(/[^\d]/g, '') === marker) ?? null;
}

/**
 * Colour-coded action badge with a tooltip.
 * @param {string} action
 */
export function actionBadge(action) {
  const meta = /** @type {Record<string, {tone: string, tooltip: string}>} */ (ACTION_META)[action];
  if (!meta) return null;
  return h(
    'span',
    { class: `badge badge--${meta.tone}`, tabindex: 0, 'data-tooltip': meta.tooltip },
    h('span', { class: 'badge__dot', 'aria-hidden': 'true' }),
    action,
    h('span', { class: 'visually-hidden', text: ` — ${meta.tooltip}` }),
  );
}

/**
 * @param {number} count
 * @param {Record<string, number> | undefined} categories
 */
function redactionChip(count, categories) {
  const kinds = categories ? Object.keys(categories).map((k) => k.toLowerCase().replace(/_/g, ' ')) : [];
  const tip =
    `${count} personal detail${count === 1 ? ' was' : 's were'} removed before your message was processed or stored` +
    (kinds.length ? ` (${kinds.join(', ')})` : '');
  return h(
    'span',
    { class: 'chip chip--privacy', tabindex: 0, 'data-tooltip': tip },
    icon('eyeOff', { size: 13 }),
    `PII redacted (${count})`,
    h('span', { class: 'visually-hidden', text: ` — ${tip}` }),
  );
}

/**
 * A user message bubble.
 * @param {Message} message
 */
export function userMessage(message) {
  return h(
    'article',
    { class: 'msg msg--user', 'aria-label': 'You said' },
    h('div', { class: 'msg__bubble', text: message.content }),
  );
}

/**
 * Sources list under an answer.
 * @param {any[]} citations
 */
function sourcesList(citations) {
  return h(
    'div',
    { class: 'sources' },
    h('p', { class: 'sources__title' }, icon('book', { size: 14 }), 'Sources'),
    h(
      'ol',
      { class: 'sources__list' },
      ...citations.map((c) => {
        const n = String(c.marker ?? '').replace(/[^\d]/g, '');
        const meta = [c.org && c.year ? `${c.org} ${c.year}` : c.org || c.year, c.section, c.page ? `p. ${c.page}` : '']
          .filter(Boolean)
          .join(' · ');
        return h(
          'li',
          null,
          h(
            'button',
            { type: 'button', class: 'source-item', 'data-marker': n, 'aria-label': `Open source ${n}: ${c.title || c.source}` },
            h('span', { class: 'source-item__n', text: n }),
            h('span', { class: 'source-item__title', text: c.title || c.source || 'Guideline' }),
            meta ? h('span', { class: 'source-item__meta', text: meta }) : null,
          ),
        );
      }),
    ),
  );
}

/**
 * Create an assistant message, initially in streaming state.
 * @param {AssistantDeps} deps
 * @param {{streaming: boolean, onRetry?: () => void, onReload?: () => void}} options
 *   `onRetry` is offered as "Try again" when the turn fails; `onReload` as
 *   "Reload conversation" when it was stopped (the server still finishes and
 *   saves a turn whose stream was disconnected).
 */
export function createAssistantMessage(deps, { streaming, onRetry, onReload }) {
  const meta = h('div', { class: 'msg__meta' });
  const content = h('div', { class: 'prose' });
  const extras = h('div', { class: 'msg__extras' });
  const actions = h('div', { class: 'msg__actions' });
  const body = h('div', { class: 'msg__body' }, meta, content, extras, actions);
  const el = h(
    'article',
    { class: `msg msg--assistant${streaming ? ' is-streaming' : ''}`, 'aria-label': `${deps.config.appName} replied` },
    h('div', { class: 'msg__avatar', 'aria-hidden': 'true' }, h('img', { src: '/static/img/logo.svg', alt: '', width: 28, height: 28 })),
    body,
  );
  const stepper = streaming ? createStepper() : null;
  if (stepper) body.insertBefore(stepper.el, content);
  if (streaming) content.append(h('span', { class: 'typing', 'aria-label': 'Preparing answer' }, h('i'), h('i'), h('i')));

  let text = '';
  let frame = 0;

  const paint = () => {
    frame = 0;
    content.innerHTML = renderMarkdown(text, { citations: false });
    content.append(h('span', { class: 'caret', 'aria-hidden': 'true' }));
  };

  /** @param {Message} message */
  const renderActions = (message) => {
    const copy = h('button', { type: 'button', class: 'icon-btn icon-btn--sm', 'aria-label': 'Copy answer', 'data-tooltip': 'Copy' }, icon('copy', { size: 16 }));
    copy.addEventListener('click', () => deps.onCopy(message));
    const regen = h('button', { type: 'button', class: 'icon-btn icon-btn--sm action-regenerate', 'aria-label': 'Regenerate answer', 'data-tooltip': 'Regenerate' }, icon('refresh', { size: 16 }));
    regen.addEventListener('click', () => deps.onRegenerate());
    const up = /** @type {HTMLButtonElement} */ (h('button', { type: 'button', class: 'icon-btn icon-btn--sm', 'aria-label': 'Good answer', 'aria-pressed': 'false', 'data-tooltip': 'Helpful' }, icon('thumbUp', { size: 16 })));
    const down = /** @type {HTMLButtonElement} */ (h('button', { type: 'button', class: 'icon-btn icon-btn--sm', 'aria-label': 'Bad answer', 'aria-pressed': 'false', 'data-tooltip': 'Not helpful' }, icon('thumbDown', { size: 16 })));
    if (!message.messageId) {
      up.disabled = true;
      down.disabled = true;
    }
    up.addEventListener('click', () => deps.onFeedback(message, 1, [up, down]));
    down.addEventListener('click', () => deps.onFeedback(message, -1, [up, down]));
    actions.replaceChildren(copy, regen, up, down);
  };

  return {
    el,
    /** @param {{stage?: string, status?: string, label?: string}} event */
    stage(event) {
      stepper?.update(event);
    },
    /** @param {{action?: string}} data */
    meta(data) {
      const badge = data && data.action ? actionBadge(String(data.action)) : null;
      if (badge) meta.replaceChildren(badge);
    },
    /** @param {string} delta */
    token(delta) {
      if (!delta) return;
      text += delta;
      if (!frame) frame = requestAnimationFrame(paint);
    },
    /**
     * Final render from a full MessageResponse.
     * @param {Message} message
     * @param {{live?: boolean}} [options]
     */
    finalize(message, { live = false } = {}) {
      if (frame) cancelAnimationFrame(frame);
      el.classList.remove('is-streaming');
      stepper?.finish({ totalMs: message.explanation?.total_ms ?? null });
      citationsByMessage.set(el, message.citations);
      el.dataset.action = message.action ?? '';

      const badges = [];
      if (message.action) badges.push(actionBadge(message.action));
      if (message.redactionCount > 0) badges.push(redactionChip(message.redactionCount, message.explanation?.redaction_categories));
      if (message.explanation?.regenerated) badges.push(h('span', { class: 'chip', text: 'Regenerated' }));
      meta.replaceChildren(...badges.filter(Boolean));

      content.innerHTML = renderMarkdown(message.content, { citations: message.citations.length > 0 });
      if (message.action === 'ESCALATE') {
        content.before(emergencyCard(deps.config, { live }));
      }
      extras.replaceChildren();
      if (message.citations.length) extras.append(sourcesList(message.citations));
      const panel = explanationPanel(message.explanation, {
        finalAction: message.action,
        open: deps.mode() === 'clinician',
      });
      if (panel) extras.append(panel);
      renderActions(message);
    },
    /**
     * Show an error / stopped state inside the message.
     * With `retryAfter` (seconds, e.g. from a 429 `Retry-After`), "Try again"
     * stays disabled and counts down; `retryText(remaining)` refreshes the notice.
     * @param {string} textMessage
     * @param {'failed' | 'stopped'} outcome
     * @param {{retryAfter?: number, retryText?: (remaining: number) => string}} [options]
     */
    fail(textMessage, outcome, { retryAfter = 0, retryText } = {}) {
      if (frame) cancelAnimationFrame(frame);
      el.classList.remove('is-streaming');
      el.classList.add(outcome === 'stopped' ? 'is-stopped' : 'is-error');
      stepper?.finish({ outcome });
      const partial = text ? renderMarkdown(text, { citations: false }) : '';
      content.innerHTML = partial;
      const noticeText = h('span', { text: textMessage });
      content.append(
        h('p', { class: `notice notice--${outcome === 'stopped' ? 'muted' : 'error'}` },
          icon(outcome === 'stopped' ? 'info' : 'alert', { size: 16 }), noticeText),
      );
      const handler = outcome === 'stopped' ? onReload : onRetry;
      if (handler) {
        const label = outcome === 'stopped' ? 'Reload conversation' : 'Try again';
        const labelNode = h('span', { text: label });
        const button = /** @type {HTMLButtonElement} */ (h('button', { type: 'button', class: 'btn btn--ghost btn--sm' }, icon('refresh', { size: 15 }), labelNode));
        button.addEventListener('click', handler, { once: true });
        actions.replaceChildren(button);
        if (outcome === 'failed' && retryAfter > 0) {
          let cancel = () => {};
          cancel = startCountdown(retryAfter, (remaining) => {
            if (!el.isConnected && remaining < retryAfter) {
              cancel(); // the message was removed (retry, new chat): stop ticking
              return;
            }
            button.disabled = remaining > 0;
            labelNode.textContent = remaining > 0 ? `${label} in ${remaining} s` : label;
            if (retryText) noticeText.textContent = retryText(remaining);
          });
        }
      }
    },
  };
}
