/**
 * Modals (focus-trapped, Escape to close when dismissible) and toasts.
 */

import { FOCUSABLE, h, trapFocus } from './dom.js';
import { icon } from './icons.js';

let modalCount = 0;

/**
 * @typedef {{label: string, variant?: 'primary' | 'danger' | 'ghost', value?: any,
 *            autofocus?: boolean, handler?: () => boolean | Promise<boolean>}} ModalAction
 *   `handler` may return false to keep the dialog open (e.g. failed validation).
 */

/**
 * Open a modal dialog. Resolves with the clicked action's `value`
 * (or `null` when dismissed).
 * @param {{title: string, body: Node | Node[], actions: ModalAction[],
 *          dismissible?: boolean, className?: string}} options
 * @returns {Promise<any>}
 */
export function openModal({ title, body, actions, dismissible = true, className = '' }) {
  const previouslyFocused = /** @type {HTMLElement | null} */ (document.activeElement);
  modalCount += 1;
  const titleId = `modal-title-${modalCount}`;

  return new Promise((resolve) => {
    const dialog = h('div', {
      class: `modal ${className}`.trim(),
      role: 'dialog',
      'aria-modal': 'true',
      'aria-labelledby': titleId,
    });
    const backdrop = h('div', { class: 'modal-backdrop' }, dialog);
    let releaseTrap = () => {};

    /** @param {any} value */
    const close = (value) => {
      releaseTrap();
      document.removeEventListener('keydown', onKeydown, true);
      backdrop.classList.add('is-leaving');
      setTimeout(() => backdrop.remove(), 120);
      document.body.classList.toggle('has-modal', document.querySelectorAll('.modal-backdrop:not(.is-leaving)').length > 0);
      if (previouslyFocused && document.contains(previouslyFocused)) previouslyFocused.focus();
      resolve(value);
    };

    /** @param {KeyboardEvent} event */
    const onKeydown = (event) => {
      if (event.key === 'Escape' && dismissible && backdrop.isConnected && !backdrop.classList.contains('is-leaving')) {
        const topmost = [...document.querySelectorAll('.modal-backdrop')].pop();
        if (topmost !== backdrop) return;
        event.stopPropagation();
        close(null);
      }
    };

    const header = h(
      'div',
      { class: 'modal__head' },
      h('h2', { class: 'modal__title', id: titleId, text: title }),
      dismissible
        ? h('button', {
            type: 'button',
            class: 'icon-btn',
            'aria-label': 'Close dialog',
            on: { click: () => close(null) },
          }, icon('close'))
        : null,
    );
    const content = h('div', { class: 'modal__body' });
    content.append(...(Array.isArray(body) ? body : [body]));
    const footer = h('div', { class: 'modal__actions' });
    for (const action of actions) {
      const button = h('button', {
        type: 'button',
        class: `btn btn--${action.variant ?? 'ghost'}`,
        text: action.label,
        'data-autofocus': action.autofocus ? 'true' : null,
        on: {
          click: async () => {
            if (action.handler && (await action.handler()) === false) return;
            close(action.value ?? null);
          },
        },
      });
      footer.append(button);
    }
    dialog.append(header, content, footer);
    if (dismissible) {
      backdrop.addEventListener('mousedown', (event) => {
        if (event.target === backdrop) close(null);
      });
    }

    document.body.append(backdrop);
    document.body.classList.add('has-modal');
    document.addEventListener('keydown', onKeydown, true);
    releaseTrap = trapFocus(dialog);
    const target =
      dialog.querySelector('[data-autofocus="true"]') ||
      content.querySelector('input, textarea') ||
      dialog.querySelector(FOCUSABLE);
    /** @type {HTMLElement | null} */ (target)?.focus();
  });
}

/**
 * Yes/no confirmation dialog.
 * @param {{title: string, message: string, confirmLabel: string, danger?: boolean}} options
 * @returns {Promise<boolean>}
 */
export async function confirmDialog({ title, message, confirmLabel, danger = false }) {
  const result = await openModal({
    title,
    body: h('p', { text: message }),
    actions: [
      { label: 'Cancel', variant: 'ghost', value: false },
      { label: confirmLabel, variant: danger ? 'danger' : 'primary', value: true, autofocus: true },
    ],
  });
  return result === true;
}

/**
 * Show a transient toast in the polite live region.
 * @param {string} message
 * @param {{tone?: 'info' | 'success' | 'error', timeout?: number}} [options]
 */
export function showToast(message, { tone = 'info', timeout = 3500 } = {}) {
  const root = document.getElementById('toasts');
  if (!root) return;
  const toast = h(
    'div',
    { class: `toast toast--${tone}` },
    icon(tone === 'error' ? 'alert' : tone === 'success' ? 'check' : 'info', { size: 16 }),
    h('span', { text: message }),
  );
  root.append(toast);
  setTimeout(() => {
    toast.classList.add('is-leaving');
    setTimeout(() => toast.remove(), 200);
  }, timeout);
}
