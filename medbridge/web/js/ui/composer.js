/**
 * Sticky, auto-growing composer: Enter sends, Shift+Enter inserts a newline,
 * 2,000-character limit with a live counter, stop button while streaming, and a
 * "hold" state (sending paused with a visible reason) while another write to the
 * same conversation is still in flight.
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
  const holdNote = byId('composer-hold');
  let busy = false;
  /** @type {string | null} */
  let hold = null;

  const update = () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 220)}px`;
    const length = input.value.length;
    counter.textContent = `${length.toLocaleString()} / ${MAX_CHARS.toLocaleString()}`;
    counter.classList.toggle('is-near', length > MAX_CHARS * 0.9);
    counter.classList.toggle('is-full', length >= MAX_CHARS);
    send.disabled = busy || hold !== null || input.value.trim().length === 0;
  };

  const submit = () => {
    const text = input.value.trim();
    if (busy || hold !== null || !text) return;
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
    /**
     * Pause sending (the typed text is kept) and show why; `null` resumes.
     * @param {string | null} reason
     */
    setHold(reason) {
      hold = reason;
      form.classList.toggle('is-held', reason !== null);
      holdNote.textContent = reason ?? '';
      holdNote.hidden = reason === null;
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
