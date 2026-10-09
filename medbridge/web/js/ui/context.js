/**
 * Patient context panel: latest BP with category, BP trend chart, manual BP
 * entry, medications, symptoms, labs, demographics, and data actions
 * (clinician summary, export, print, clear my data).
 *
 * The snapshot shown here is the server's event-sourced projection; manual
 * readings go through POST /bp so they follow the same event pipeline. A
 * reading is never saved while another write to the same conversation (an
 * answer being generated or finished) is in flight: the form is disabled and
 * says why (see writeguard.js).
 */

import { BP_LIMITS, categoryLabel, validateBpInput } from '../bp.js';
import { errorMessage } from '../errors.js';
import { formatDateTime, toLocalInputValue } from '../format.js';
import { normalizeContext } from '../normalize.js';
import { renderBpChart } from './bpchart.js';
import { byId, h } from './dom.js';
import { icon } from './icons.js';

/**
 * @typedef {{
 *   api: ReturnType<typeof import('../api.js').createApi>,
 *   getSession: () => import('../store.js').Conversation | null,
 *   ensureSession: (title: string) => Promise<import('../store.js').Conversation | null>,
 *   onCrisis: (reading: {systolic: number, diastolic: number}, advice: string | null) => void,
 *   onSummary: () => void, onExport: () => void, onPrint: () => void, onClear: () => void,
 *   onToast: (message: string, tone?: 'info' | 'success' | 'error') => void,
 *   onVisibility: (open: boolean) => void,
 *   writeBlocked: (sessionId: string | null) => string | null,
 *   beginWrite: (sessionId: string) => {release: () => void} | null,
 * }} ContextDeps
 *   `writeBlocked` returns why writes to the conversation are paused (or null);
 *   `beginWrite` takes the conversation's write lock (null when it is held).
 */

/**
 * @param {ContextDeps} deps
 */
export function createContextPanel(deps) {
  const panel = byId('context-panel');
  const body = byId('context-body');
  let open = false;
  let loadSeq = 0;

  /**
   * One labelled number input of the BP form.
   * @param {string} name
   * @param {string} label
   * @param {{min: number, max: number}} limits
   * @param {boolean} required
   */
  const numberField = (name, label, limits, required) =>
    h('div', { class: 'field' },
      h('label', { for: `bp-${name}`, text: label }),
      h('input', { id: `bp-${name}`, name, type: 'number', inputmode: 'numeric', min: limits.min, max: limits.max, step: 1, required, 'aria-describedby': `bp-${name}-err` }),
      h('p', { class: 'field__error', id: `bp-${name}-err`, 'aria-live': 'polite' }));

  const bpForm = () => {
    const form = /** @type {HTMLFormElement} */ (h('form', { class: 'bp-form', novalidate: true },
      h('div', { class: 'bp-form__grid' },
        numberField('systolic', 'Systolic (top)', BP_LIMITS.systolic, true),
        numberField('diastolic', 'Diastolic (bottom)', BP_LIMITS.diastolic, true),
        numberField('pulse', 'Pulse (optional)', BP_LIMITS.pulse, false),
        h('div', { class: 'field' },
          h('label', { for: 'bp-measuredAt', text: 'Measured at' }),
          h('input', { id: 'bp-measuredAt', name: 'measuredAt', type: 'datetime-local', value: toLocalInputValue(new Date()), 'aria-describedby': 'bp-measuredAt-err' }),
          h('p', { class: 'field__error', id: 'bp-measuredAt-err', 'aria-live': 'polite' }))),
      h('p', { class: 'bp-form__hold', role: 'status', hidden: true }),
      h('button', { type: 'submit', class: 'btn btn--primary btn--block' }, icon('plus', { size: 16 }), 'Add reading')));
    const when = /** @type {HTMLInputElement} */ (form.querySelector('#bp-measuredAt'));
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const data = Object.fromEntries(new FormData(form).entries());
      // Untouched prefilled time (minute precision) means "now", to the second.
      if (data.measuredAt === when.defaultValue) data.measuredAt = '';
      const result = validateBpInput(/** @type {any} */ (data));
      for (const name of ['systolic', 'diastolic', 'pulse', 'measuredAt']) {
        const err = /** @type {HTMLElement} */ (form.querySelector(`#bp-${name}-err`));
        err.textContent = result.errors[name] ?? '';
        form.querySelector(`#bp-${name}`)?.setAttribute('aria-invalid', result.errors[name] ? 'true' : 'false');
      }
      if (!result.ok) {
        /** @type {HTMLElement | null} */ (form.querySelector('[aria-invalid="true"]'))?.focus();
        return;
      }
      const blocked = deps.writeBlocked(deps.getSession()?.id ?? null);
      if (blocked) {
        deps.onToast(blocked, 'info');
        return;
      }
      form.dataset.submitting = 'true';
      syncWriteState();
      /** @type {{release: () => void} | null} */
      let lock = null;
      try {
        const session = deps.getSession() ?? (await deps.ensureSession('Blood pressure log'));
        if (!session) return;
        lock = deps.beginWrite(session.id);
        if (!lock) {
          deps.onToast(deps.writeBlocked(session.id) ?? 'Please wait a moment and try again.', 'info');
          return;
        }
        const response = await deps.api.postBp(session, result.value);
        when.defaultValue = toLocalInputValue(new Date());
        form.reset();
        deps.onToast(`Reading ${result.value.systolic}/${result.value.diastolic} saved`, 'success');
        if (response && response.crisis === true) {
          deps.onCrisis(result.value, typeof response.advice === 'string' ? response.advice : null);
        }
        await refresh();
      } catch (error) {
        deps.onToast(errorMessage(/** @type {any} */ (error), 'Could not save the reading.'), 'error');
      } finally {
        delete form.dataset.submitting;
        lock?.release();
        syncWriteState();
      }
    });
    return form;
  };

  /** Enable/disable every BP form for the open conversation's write state. */
  function syncWriteState() {
    const reason = deps.writeBlocked(deps.getSession()?.id ?? null);
    for (const form of body.querySelectorAll('form.bp-form')) {
      const submitting = /** @type {HTMLElement} */ (form).dataset.submitting === 'true';
      const button = /** @type {HTMLButtonElement | null} */ (form.querySelector('button[type="submit"]'));
      const note = /** @type {HTMLElement | null} */ (form.querySelector('.bp-form__hold'));
      if (button) button.disabled = submitting || reason !== null;
      if (note) {
        note.textContent = submitting ? '' : reason ?? '';
        note.hidden = submitting || reason === null;
      }
    }
  }

  /** @param {string} title @param {...(Node | null)} children */
  const card = (title, ...children) => h('section', { class: 'ctx-card' }, h('h3', { class: 'ctx-card__title', text: title }), ...children);

  const actionsCard = (hasSession) => {
    const button = (label, iconName, handler, variant = 'ghost') => {
      const b = h('button', { type: 'button', class: `btn btn--${variant} btn--sm`, disabled: !hasSession }, icon(iconName, { size: 15 }), label);
      b.addEventListener('click', handler);
      return b;
    };
    return h('section', { class: 'ctx-actions', 'aria-label': 'Your data' },
      button('Clinician summary', 'download', deps.onSummary),
      button('Export conversation', 'file', deps.onExport),
      button('Print / Save as PDF', 'printer', deps.onPrint),
      button('Clear my data', 'trash', deps.onClear, 'danger-ghost'));
  };

  /** @param {ReturnType<typeof normalizeContext>} ctx */
  const renderContext = (ctx) => {
    const nodes = [];
    if (ctx.latest) {
      const r = ctx.latest;
      nodes.push(h('section', { class: 'ctx-card ctx-latest' },
        h('p', { class: 'ctx-card__title', text: 'Latest blood pressure' }),
        h('div', { class: 'ctx-latest__row' },
          h('p', { class: 'ctx-latest__value' }, `${r.systolic}/${r.diastolic}`, h('span', { class: 'ctx-latest__unit', text: ' mmHg' })),
          h('span', { class: `cat-pill cat-${r.category}`, text: categoryLabel(/** @type {any} */ (r.category)) })),
        h('p', { class: 'ctx-latest__meta', text: [r.pulse ? `Pulse ${r.pulse}` : '', formatDateTime(r.time)].filter(Boolean).join(' · ') || ' ' })));
    }
    if (ctx.series.length) {
      const chart = h('div', { class: 'ctx-chart' });
      nodes.push(card('Blood pressure trend', chart));
      requestAnimationFrame(() => renderBpChart(chart, ctx.series));
    }
    nodes.push(h('section', { class: 'ctx-card' },
      h('details', { class: 'ctx-form', open: !ctx.series.length },
        h('summary', null, icon('plus', { size: 16 }), 'Add a blood pressure reading'), bpForm())));
    if (ctx.medications.length) {
      nodes.push(card('Medications', h('ul', { class: 'ctx-list' }, ...ctx.medications.map((m) =>
        h('li', null, h('span', { class: 'ctx-list__name', text: m.name }), m.detail ? h('span', { class: 'ctx-list__detail', text: m.detail }) : null)))));
    }
    if (ctx.symptoms.length) {
      nodes.push(card('Symptoms mentioned', h('div', { class: 'chips' }, ...ctx.symptoms.map((s) => h('span', { class: 'chip', text: s })))));
    }
    if (ctx.labs.length) {
      nodes.push(card('Lab results', h('dl', { class: 'kv' }, ...ctx.labs.flatMap((l) => [h('dt', { text: l.name }), h('dd', { text: l.value || '—' })]))));
    }
    if (ctx.demographics.length) {
      nodes.push(card('About you', h('dl', { class: 'kv' }, ...ctx.demographics.flatMap(([k, v]) => [h('dt', { text: k }), h('dd', { text: v })]))));
    }
    if (ctx.isEmpty) {
      nodes.unshift(h('div', { class: 'ctx-empty' }, icon('pulse', { size: 22 }),
        h('p', { text: 'Nothing recorded yet. Readings, medicines and symptoms you mention in the chat will appear here.' })));
    }
    return nodes;
  };

  async function refresh() {
    if (!open) return;
    const seq = ++loadSeq;
    const session = deps.getSession();
    const note = h('p', { class: 'ctx-note' }, icon('lock', { size: 14 }),
      'Built only from what you share here. Stored encrypted; you can erase it at any time.');
    if (!session) {
      body.replaceChildren(
        h('div', { class: 'ctx-empty' }, icon('pulse', { size: 22 }),
          h('p', { text: 'Start a conversation or add a reading to build your health context.' })),
        h('section', { class: 'ctx-card' }, h('details', { class: 'ctx-form', open: true },
          h('summary', null, icon('plus', { size: 16 }), 'Add a blood pressure reading'), bpForm())),
        actionsCard(false), note);
      syncWriteState();
      return;
    }
    body.setAttribute('aria-busy', 'true');
    try {
      const ctx = normalizeContext(await deps.api.getContext(session));
      if (seq !== loadSeq) return;
      body.replaceChildren(...renderContext(ctx), actionsCard(true), note);
      syncWriteState();
    } catch (error) {
      if (seq !== loadSeq) return;
      body.replaceChildren(
        h('p', { class: 'notice notice--error' }, icon('alert', { size: 16 }),
          /** @type {any} */ (error).sessionUnavailable
            ? errorMessage(/** @type {any} */ (error))
            : 'Could not load your health context. Please try again.'),
        actionsCard(true), note);
    } finally {
      body.removeAttribute('aria-busy');
    }
  }

  /** @param {boolean} value */
  function setOpen(value) {
    open = value;
    panel.hidden = !value;
    panel.classList.toggle('is-open', value);
    deps.onVisibility(value);
    if (value) {
      refresh();
      byId('context-close').focus();
    }
  }

  byId('context-close').addEventListener('click', () => setOpen(false));

  return {
    get isOpen() {
      return open;
    },
    toggle() {
      setOpen(!open);
    },
    close() {
      if (open) setOpen(false);
    },
    refresh,
    syncWriteState,
  };
}
