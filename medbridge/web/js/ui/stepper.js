/**
 * Live pipeline stepper fed by SSE `stage` events.
 *
 * Shows each stage as it starts ("Checking for emergencies…" → … →
 * "Writing answer…"); skipped stages are greyed. When the answer is final the
 * stepper collapses into a one-line summary that can be re-opened.
 */

import { STAGE_LABELS } from '../actions.js';
import { formatMs } from '../format.js';
import { h } from './dom.js';
import { icon } from './icons.js';

/**
 * @typedef {{stage?: string, status?: string, label?: string}} StageEvent
 */

export function createStepper() {
  const label = h('span', { class: 'steps__label', text: 'Starting…' });
  const indicator = h('span', { class: 'steps__indicator', 'aria-hidden': 'true' }, h('span', { class: 'spinner' }));
  const list = h('ol', { class: 'steps__list' });
  const el = h('details', { class: 'steps is-running', open: true }, h('summary', null, indicator, label), list);
  /** @type {Map<string, HTMLLIElement>} */
  const items = new Map();
  const started = performance.now();

  /**
   * @param {HTMLLIElement} li
   * @param {string} status
   */
  const setStatus = (li, status) => {
    li.dataset.status = status;
    const slot = /** @type {HTMLElement} */ (li.querySelector('.step__icon'));
    if (status === 'start') slot.replaceChildren(h('span', { class: 'spinner spinner--sm' }));
    else if (status === 'skipped') slot.replaceChildren(h('span', { class: 'step__dash' }));
    else slot.replaceChildren(icon('check', { size: 14 }));
  };

  return {
    el,
    /** @param {StageEvent} event */
    update(event) {
      const stage = String(event.stage ?? '');
      if (!stage) return;
      const status = event.status === 'skipped' ? 'skipped' : event.status === 'end' ? 'end' : 'start';
      const text = event.label || /** @type {Record<string, string>} */ (STAGE_LABELS)[stage] || stage;
      let li = items.get(stage);
      if (!li) {
        li = h('li', { class: 'step' }, h('span', { class: 'step__icon', 'aria-hidden': 'true' }), h('span', { class: 'step__text' }));
        items.set(stage, li);
        list.append(li);
      }
      /** @type {HTMLElement} */ (li.querySelector('.step__text')).textContent =
        status === 'skipped' ? `${text.replace(/…$/, '')} (skipped)` : text;
      setStatus(li, status);
      if (status === 'start') label.textContent = text;
    },
    /**
     * Collapse into a summary line.
     * @param {{totalMs?: number | null, outcome?: 'done' | 'stopped' | 'failed'}} [info]
     */
    finish({ totalMs = null, outcome = 'done' } = {}) {
      for (const li of items.values()) if (li.dataset.status === 'start') setStatus(li, outcome === 'done' ? 'end' : 'skipped');
      const done = [...items.values()].filter((li) => li.dataset.status === 'end').length;
      const elapsed = totalMs ?? performance.now() - started;
      el.classList.remove('is-running');
      el.open = false;
      indicator.replaceChildren(icon(outcome === 'done' ? 'check' : 'alert', { size: 14 }));
      if (outcome === 'stopped') label.textContent = 'Stopped';
      else if (outcome === 'failed') label.textContent = 'Something went wrong';
      else label.textContent = `Checked in ${done} step${done === 1 ? '' : 's'} · ${formatMs(elapsed)}`;
      if (!items.size) el.remove();
    },
  };
}
