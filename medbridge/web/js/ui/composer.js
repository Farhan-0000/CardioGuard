/**
 * Sticky, auto-growing composer: Enter sends, Shift+Enter inserts a newline,
 * 2,000-character limit with a live counter, stop button while streaming.
 */

import { byId } from './dom.js';

export const MAX_CHARS = 2000;

/**
 * @param {{onSubmit: (text: string) => void, onStop: () => void}} handlers
 */
export function createComposer({ onSubmit, onStop }) {
  const form = /** @type {HTMLFormElement} */ (byId('composer'));
  const input = /** @type {HTMLTextAreaElement} */ (byId('composer-input'));
  const counter = byId('char-count');
  const send = /** @type {HTMLButtonElement} */ (byId('send-btn'));
  const stop = /** @type {HTMLButtonElement} */ (byId('stop-btn'));
  let busy = false;

  const update = () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 220)}px`;
    const length = input.value.length;
    counter.textContent = `${length.toLocaleString()} / ${MAX_CHARS.toLocaleString()}`;
    counter.classList.toggle('is-near', length > MAX_CHARS * 0.9);
    counter.classList.toggle('is-full', length >= MAX_CHARS);
    send.disabled = busy || input.value.trim().length === 0;
  };

  const submit = () => {
    const text = input.value.trim();
    if (busy || !text) return;
    if (text.length > MAX_CHARS) return;
    input.value = '';
    update();
    onSubmit(text);
  };

  input.addEventListener('input', update);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    }
  });
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    submit();
  });
  stop.addEventListener('click', () => onStop());
  update();

  return {
    /** @param {boolean} value */
    setBusy(value) {
      busy = value;
      form.classList.toggle('is-busy', value);
      send.hidden = value;
      stop.hidden = !value;
      update();
    },
    focus() {
      input.focus();
    },
    /** @param {string} text */
    setText(text) {
      input.value = text.slice(0, MAX_CHARS);
      update();
      input.focus();
    },
  };
}
