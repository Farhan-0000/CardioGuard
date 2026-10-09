/**
 * Admin / evaluation dashboard controller.
 *
 * Login → POST /api/admin/login → token kept in sessionStorage (tab-scoped,
 * try/catch with in-memory fallback). Metrics from GET /api/metrics and the
 * latest audit rows from GET /api/admin/audit?limit=50, both with the bearer
 * token. All values are rendered as text; charts are hand-written SVG.
 */

import { ACTION_META, ACTIONS, STAGE_LABELS, normalizeAction, stageShortName } from './actions.js';
import { ApiError, createApi } from './api.js';
import { formatCount, formatDateTime, formatMs, formatPercent, stripMarkdown } from './format.js';
import { normalizeAudit, normalizeMetrics } from './normalize.js';
import { createSafeStorage, createStore, resolveStorage } from './store.js';
import { dataTable, renderBarChart } from './ui/charts.js';
import { byId, h } from './ui/dom.js';
import { hydrateIcons } from './ui/icons.js';
import { actionBadge } from './ui/messages.js';
import { showToast } from './ui/overlay.js';
import { initThemeToggle } from './ui/theme.js';

const api = createApi();
const session = createSafeStorage(resolveStorage('sessionStorage'));
const TOKEN_KEY = 'adminToken';
const AUDIT_LIMIT = 50;

const view = {
  login: byId('login-view'),
  metrics: byId('metrics-view'),
  form: /** @type {HTMLFormElement} */ (byId('login-form')),
  error: byId('login-error'),
  submit: /** @type {HTMLButtonElement} */ (byId('login-submit')),
  refresh: byId('refresh'),
  logout: byId('logout'),
  updated: byId('updated'),
};

/** @param {boolean} signedIn */
function showSignedIn(signedIn) {
  view.login.hidden = signedIn;
  view.metrics.hidden = !signedIn;
  view.refresh.hidden = !signedIn;
  view.logout.hidden = !signedIn;
  if (!signedIn) view.updated.textContent = '';
}

/** Tone for a decision/action key. */
function toneOf(key) {
  const action = normalizeAction(key);
  return action ? ACTION_META[action].tone : String(key).toUpperCase() === 'PROCEED' ? 'proceed' : 'series';
}

/**
 * Bars for a count distribution with percentages.
 * @param {HTMLElement} container
 * @param {[string, number][]} entries
 * @param {string} label
 * @param {string[]} [order]
 */
function countChart(container, entries, label, order = []) {
  const sorted = [...entries].sort((a, b) => {
    const ia = order.indexOf(normalizeAction(a[0]) ?? a[0]);
    const ib = order.indexOf(normalizeAction(b[0]) ?? b[0]);
    if (ia !== -1 || ib !== -1) return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    return b[1] - a[1];
  });
  const total = sorted.reduce((sum, [, n]) => sum + n, 0);
  const chart = h('div');
  container.replaceChildren(
    chart,
    dataTable(label, ['Decision', 'Count', 'Share'], sorted.map(([k, n]) => [k, n, formatPercent(total ? n / total : 0)])),
  );
  renderBarChart(chart, {
    ariaLabel: label,
    rows: sorted.map(([key, n]) => ({
      label: key,
      value: n,
      tone: toneOf(key),
      valueLabel: `${formatCount(n)} · ${formatPercent(total ? n / total : 0)}`,
    })),
  });
}

/**
 * Sort rows by a canonical key order (unknown keys last, alphabetically).
 * @template {{key: string}} T
 * @param {T[]} rows
 * @param {string[]} order
 * @returns {T[]}
 */
function inOrder(rows, order) {
  const rank = (key) => {
    const i = order.indexOf(key);
    return i === -1 ? order.length : i;
  };
  return [...rows].sort((a, b) => rank(a.key) - rank(b.key) || a.key.localeCompare(b.key));
}

/**
 * p50 bar over a p95 track.
 * @param {HTMLElement} container
 * @param {{key: string, p50: number | null, p95: number | null, n: number | null}[]} rows
 * @param {string} label
 * @param {(key: string) => string} name
 */
function latencyChart(container, rows, label, name) {
  const chart = h('div');
  container.replaceChildren(
    chart,
    dataTable(label, ['Group', 'p50', 'p95', 'n'], rows.map((r) => [name(r.key), formatMs(r.p50), formatMs(r.p95), r.n ?? '—'])),
  );
  renderBarChart(chart, {
    ariaLabel: label,
    rows: rows.map((r) => ({
      label: name(r.key),
      value: r.p50 ?? 0,
      track: r.p95,
      valueLabel: `${formatMs(r.p50)} · ${formatMs(r.p95)}`,
      tooltip: `${name(r.key)}: p50 ${formatMs(r.p50)}, p95 ${formatMs(r.p95)}${r.n ? ` (n = ${r.n})` : ''}`,
    })),
  });
}

/**
 * @param {string} label
 * @param {string} value
 * @param {string} note
 */
function kpi(label, value, note) {
  return h('div', { class: 'kpi' }, h('p', { class: 'kpi__label', text: label }), h('p', { class: 'kpi__value', text: value }), h('p', { class: 'kpi__note', text: note }));
}

/**
 * "Decided by: llm 40 · fallback 1" summary of gate decision sources.
 * @param {[string, number][]} entries
 */
function sourcesLine(entries) {
  if (!entries.length) return '';
  return `Decided by: ${entries.map(([k, n]) => `${k.replace(/_/g, ' ')} ${formatCount(n)}`).join(' · ')}`;
}

/** @param {{title: string, entries: [string, number][]}[]} groups */
function renderSafety(groups) {
  const root = byId('safety');
  if (!groups.length) {
    root.replaceChildren(h('p', { class: 'chart__empty', text: 'No safety-mechanism counters reported yet.' }));
    return;
  }
  root.replaceChildren(
    ...groups.map((g) =>
      h('section', { class: 'safety-group' },
        h('h3', { class: 'safety-group__title', text: g.title }),
        h('dl', { class: 'kv' }, ...g.entries.flatMap(([k, v]) => [
          h('dt', { text: k }),
          h('dd', { text: Number.isInteger(v) ? formatCount(v) : v.toFixed(2) }),
        ])))),
  );
}

/** @param {ReturnType<typeof normalizeMetrics>} m */
function renderMetrics(m) {
  byId('kpis').replaceChildren(
    kpi('Total turns', formatCount(m.totalTurns), m.windowLabel || 'All recorded turns'),
    kpi('Median latency (p50)', formatMs(m.p50), 'End-to-end per turn'),
    kpi('p95 latency', formatMs(m.p95), '95th percentile per turn'),
    kpi('Thumbs-up rate', formatPercent(m.feedback.rate), `${formatCount(m.feedback.up + m.feedback.down)} rated answers`),
  );
  countChart(byId('chart-actions'), m.actions, 'Action distribution', [...ACTIONS]);
  countChart(byId('chart-gate1'), m.gate1, 'Gate 1 decisions', ['PROCEED', 'SOFT-ASK']);
  countChart(byId('chart-gate2'), m.gate2, 'Gate 2 decisions', [...ACTIONS]);
  byId('sources-gate1').textContent = sourcesLine(m.gate1Sources);
  byId('sources-gate2').textContent = sourcesLine(m.gate2Sources);
  renderSafety(m.safety);
  latencyChart(byId('chart-latency-action'), inOrder(m.byAction, [...ACTIONS]), 'Latency by action', (k) => normalizeAction(k) ?? k);
  latencyChart(byId('chart-latency-stage'), inOrder(m.byStage, Object.keys(STAGE_LABELS)), 'Latency by stage', stageShortName);

  const chart = h('div');
  byId('feedback').replaceChildren(
    h('div', { class: 'feedback' },
      h('p', { class: 'feedback__rate' }, formatPercent(m.feedback.rate), h('small', { text: 'thumbs-up rate' })),
      chart),
  );
  renderBarChart(chart, {
    ariaLabel: 'Feedback counts',
    rows: [
      { label: '👍 Helpful', value: m.feedback.up, tone: 'answer', valueLabel: formatCount(m.feedback.up) },
      { label: '👎 Not helpful', value: m.feedback.down, tone: 'escalate', valueLabel: formatCount(m.feedback.down) },
    ],
  });
}

/** @param {ReturnType<typeof normalizeAudit>} rows */
function renderAudit(rows) {
  const table = byId('audit');
  const head = h('thead', null, h('tr', null, ...['Time', 'Session', 'Action', 'Gate 1', 'Gate 2', 'Latency', 'Request (redacted)', 'Response (redacted)']
    .map((t) => h('th', { scope: 'col', text: t }))));
  if (!rows.length) {
    table.replaceChildren(head, h('tbody', null, h('tr', null, h('td', { colspan: 8, class: 'audit-empty', text: 'No audit entries yet.' }))));
    return;
  }
  table.replaceChildren(
    head,
    h('tbody', null, ...rows.map((r) => h('tr', null,
      h('td', { class: 'when', text: formatDateTime(r.time) }),
      h('td', { class: 'ref', text: r.session || '—' }),
      h('td', null, r.action ? actionBadge(r.action) : '—'),
      h('td', { class: 'gate', text: r.gate1 || '—' }),
      h('td', { class: 'gate', text: r.gate2 || '—' }),
      h('td', { class: 'num', text: formatMs(r.latency) }),
      h('td', { class: 'text', title: r.request }, h('span', { text: r.erased ? '[erased]' : stripMarkdown(r.request) || '—' })),
      h('td', { class: 'text', title: r.response }, h('span', { text: r.erased ? '[erased]' : stripMarkdown(r.response) || '—' })),
    ))),
  );
}

async function load() {
  const token = session.get(TOKEN_KEY);
  if (!token) {
    showSignedIn(false);
    return;
  }
  showSignedIn(true);
  try {
    const [metrics, audit] = await Promise.all([api.getMetrics(token), api.getAudit(token, AUDIT_LIMIT)]);
    renderMetrics(normalizeMetrics(metrics));
    renderAudit(normalizeAudit(audit));
    view.updated.textContent = `Updated ${new Date().toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`;
  } catch (error) {
    const err = /** @type {ApiError} */ (error);
    if (err.status === 401 || err.status === 403) {
      session.remove(TOKEN_KEY);
      showSignedIn(false);
      view.error.textContent = 'Your admin session has expired. Please sign in again.';
      view.error.hidden = false;
      return;
    }
    showToast(err.message || 'Could not load metrics.', { tone: 'error' });
  }
}

view.form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const data = new FormData(view.form);
  const username = String(data.get('username') ?? '').trim();
  const password = String(data.get('password') ?? '');
  view.error.hidden = true;
  if (!username || !password) {
    view.error.textContent = 'Enter your username and password.';
    view.error.hidden = false;
    return;
  }
  view.submit.disabled = true;
  try {
    const result = await api.adminLogin(username, password);
    const token = String(result?.token ?? result?.access_token ?? '');
    if (!token) throw new ApiError('No token returned', { status: 500 });
    session.set(TOKEN_KEY, token);
    view.form.reset();
    await load();
  } catch (error) {
    const err = /** @type {ApiError} */ (error);
    view.error.textContent =
      err.status === 401 ? 'Incorrect username or password.'
        : err.status === 403 ? 'The dashboard is disabled on this server (ADMIN_PASSWORD is not set).'
          : err.status === 429 ? 'Too many attempts. Please wait a minute and try again.'
            : err.status === 422 ? 'Enter a valid username and password.'
              : 'Sign-in failed. Is the MedBridge server running?';
    view.error.hidden = false;
  } finally {
    view.submit.disabled = false;
  }
});

view.refresh.addEventListener('click', () => load());
view.logout.addEventListener('click', () => {
  session.remove(TOKEN_KEY);
  showSignedIn(false);
  /** @type {HTMLElement} */ (byId('username')).focus();
});

hydrateIcons(document);
initThemeToggle(byId('theme-toggle'), createStore());
load();
