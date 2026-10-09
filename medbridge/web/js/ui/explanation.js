/**
 * "Why this answer?" panel — the Dual Answerability Gate made visible.
 *
 * Renders the `Explanation` model of a MessageResponse: decision path,
 * Gate 1 (context) decision + rationale + missing fields, the top-5 reranked
 * evidence, Gate 2 (evidence) decision + rationale (+ votes/features for the
 * calibrated variant), Gate 3 grounding summary, emergency rules, citation
 * validation and a per-stage latency timeline. All text is set as textContent.
 */

import { normalizeAction, stageShortName } from '../actions.js';
import { formatMs } from '../format.js';
import { renderBarChart } from './charts.js';
import { h } from './dom.js';
import { icon } from './icons.js';

const SOURCE_LABELS = {
  llm: 'Model decision (temperature 0, seed 42)',
  fallback: 'Safe fallback — model unavailable',
  loop_breaker: 'Loop-breaker — asked twice already',
  fastpath: 'Deterministic emergency rule',
  rule: 'Deterministic rule',
  calibrated: 'Calibrated: evidence signal + votes',
  ablation: 'Disabled for evaluation',
};

/** @param {unknown} v */
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/**
 * Tone class for a gate decision pill.
 * @param {string} decision
 */
function toneFor(decision) {
  const action = normalizeAction(decision);
  if (action === 'ANSWER') return 'answer';
  if (action === 'GENERALIZE') return 'generalize';
  if (action === 'SOFT-ASK') return 'softask';
  if (action === 'ESCALATE') return 'escalate';
  if (action === 'ABSTAIN') return 'abstain';
  if (String(decision).toUpperCase() === 'PROCEED') return 'proceed';
  return 'neutral';
}

/**
 * @param {string} decision
 * @param {string} [prefix]
 */
function pill(decision, prefix = '') {
  return h('span', { class: `pill pill--${toneFor(decision)}` }, prefix ? h('span', { class: 'pill__prefix', text: prefix }) : null, decision);
}

/**
 * Format a feature value for display.
 * @param {unknown} value
 */
function featureValue(value) {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (Array.isArray(value)) return value.map(String).join(', ');
  if (isObject(value)) return JSON.stringify(value);
  return String(value ?? '');
}

/**
 * Card for one gate decision.
 * @param {string} title
 * @param {string} subtitle
 * @param {any} gate - GateDecision
 */
function gateCard(title, subtitle, gate) {
  const card = h(
    'section',
    { class: 'why-card' },
    h(
      'div',
      { class: 'why-card__head' },
      h('div', null, h('h4', { text: title }), h('p', { class: 'why-card__sub', text: subtitle })),
      pill(String(gate.action || '—')),
    ),
  );
  const source = /** @type {Record<string, string>} */ (SOURCE_LABELS)[gate.source] ?? gate.source;
  if (source) card.append(h('p', { class: 'why-card__source' }, icon('info', { size: 14 }), source));
  card.append(h('p', { class: 'why-card__rationale', text: gate.rationale || 'No rationale recorded.' }));
  if (Array.isArray(gate.missing_fields) && gate.missing_fields.length) {
    card.append(
      h('div', { class: 'why-card__row' }, h('span', { class: 'why-card__label', text: 'Missing' }),
        h('div', { class: 'chips' }, ...gate.missing_fields.map((f) => h('span', { class: 'chip', text: String(f).replace(/_/g, ' ') })))),
    );
  }
  if (Array.isArray(gate.votes) && gate.votes.length) {
    card.append(
      h('div', { class: 'why-card__row' }, h('span', { class: 'why-card__label', text: 'Votes' }),
        h('div', { class: 'chips' }, ...gate.votes.map((v) => pill(String(v))))),
    );
  }
  if (isObject(gate.features) && Object.keys(gate.features).length) {
    card.append(
      h('dl', { class: 'kv' }, ...Object.entries(gate.features).flatMap(([k, v]) => [
        h('dt', { text: k.replace(/_/g, ' ') }),
        h('dd', { text: featureValue(v) }),
      ])),
    );
  }
  return card;
}

/**
 * Relevance bar for a cross-encoder score (logit → 0..1 via the logistic function).
 * @param {number} score
 */
function scoreBar(score) {
  const pct = Math.round(100 / (1 + Math.exp(-score)));
  return h(
    'span',
    { class: 'score', title: `Rerank score ${score.toFixed(2)} (relevance ≈ ${pct}%)` },
    h('span', { class: 'score__track' }, h('span', { class: 'score__fill', style: { width: `${pct}%` } })),
    h('span', { class: 'score__num', text: score.toFixed(2) }),
  );
}

/**
 * @param {any[]} retrieval - RetrievedEvidence[]
 */
function evidenceSection(retrieval) {
  return h(
    'section',
    { class: 'why-card' },
    h('div', { class: 'why-card__head' }, h('div', null, h('h4', { text: 'Evidence retrieved' }),
      h('p', { class: 'why-card__sub', text: 'Hybrid dense + BM25 search → RRF (k = 60) → cross-encoder rerank → top 5' }))),
    h(
      'ol',
      { class: 'evidence' },
      ...retrieval.map((item) =>
        h(
          'li',
          { class: `evidence__item${item.cited ? ' is-cited' : ''}` },
          h('span', { class: 'evidence__rank', text: String(item.rank ?? '') }),
          h(
            'div',
            { class: 'evidence__main' },
            h('p', { class: 'evidence__title', text: item.title || item.guideline_id || item.chunk_id }),
            h('p', { class: 'evidence__meta', text: [item.org, item.year, item.section, item.page ? `p. ${item.page}` : ''].filter(Boolean).join(' · ') }),
          ),
          scoreBar(Number(item.score_rerank) || 0),
          item.cited ? h('span', { class: 'tag tag--cited', text: 'Cited' }) : h('span', { class: 'tag tag--muted', text: 'Not cited' }),
        ),
      ),
    ),
  );
}

/**
 * @param {Record<string, any>} gate3
 */
function groundingSection(gate3) {
  const claims = Number(gate3.n_claims ?? 0);
  const unsupported = Number(gate3.n_unsupported ?? 0);
  const fraction = Number(gate3.unsupported_fraction);
  const card = h(
    'section',
    { class: 'why-card' },
    h('div', { class: 'why-card__head' }, h('div', null, h('h4', { text: 'Gate 3 · Grounding check' }),
      h('p', { class: 'why-card__sub', text: 'Each sentence checked against the cited evidence (NLI)' })),
      gate3.action ? pill(String(gate3.action)) : null),
    h('p', { class: 'why-card__rationale', text:
      `${claims - unsupported} of ${claims} claim${claims === 1 ? '' : 's'} supported` +
      (Number.isFinite(fraction) ? ` · ${Math.round(fraction * 100)}% unsupported` : '') }),
  );
  const flagged = (Array.isArray(gate3.sentences) ? gate3.sentences : []).filter((s) => isObject(s) && s.supported === false).slice(0, 5);
  if (flagged.length) {
    card.append(h('ul', { class: 'why-list' }, ...flagged.map((s) => h('li', { text: String(s.text ?? '') }))));
  }
  return card;
}

/**
 * @param {Record<string, any>} emergency
 */
function emergencySection(emergency) {
  const rules = emergency.matched_rules ?? emergency.rules ?? emergency.matches ?? emergency.patterns ?? [];
  const items = (Array.isArray(rules) ? rules : [rules]).map((r) => (isObject(r) ? r.rule ?? r.name ?? r.label ?? JSON.stringify(r) : String(r))).filter(Boolean);
  const scalars = Object.entries(emergency).filter(([k, v]) => !['matched_rules', 'rules', 'matches', 'patterns'].includes(k) && (typeof v !== 'object' || v === null));
  return h(
    'section',
    { class: 'why-card' },
    h('div', { class: 'why-card__head' }, h('div', null, h('h4', { text: 'Emergency fast-path' }),
      h('p', { class: 'why-card__sub', text: 'Deterministic rules, evaluated before any model call' }))),
    items.length ? h('div', { class: 'chips' }, ...items.map((t) => h('span', { class: 'chip chip--danger', text: t }))) : null,
    scalars.length ? h('dl', { class: 'kv' }, ...scalars.flatMap(([k, v]) => [h('dt', { text: k.replace(/_/g, ' ') }), h('dd', { text: featureValue(v) })])) : null,
  );
}

/**
 * Decision path summary: emergency → Gate 1 → retrieval → Gate 2 → final.
 * @param {any} exp
 * @param {string | null} finalAction
 */
function decisionPath(exp, finalAction) {
  const steps = [];
  const fastpath = finalAction === 'ESCALATE' && !exp.gate1;
  const emergencyMatched = isObject(exp.emergency)
    ? Boolean(exp.emergency.matched ?? exp.emergency.escalate ?? fastpath)
    : fastpath;
  steps.push(h('span', { class: `path__step${emergencyMatched ? ' is-alert' : ''}` }, h('span', { class: 'path__k', text: 'Emergency' }), emergencyMatched ? 'matched' : 'clear'));
  if (exp.gate1) steps.push(h('span', { class: 'path__step' }, h('span', { class: 'path__k', text: 'Gate 1' }), pill(String(exp.gate1.action))));
  else steps.push(h('span', { class: 'path__step is-skipped' }, h('span', { class: 'path__k', text: 'Gate 1' }), 'skipped'));
  const n = Array.isArray(exp.retrieval) ? exp.retrieval.length : 0;
  steps.push(h('span', { class: `path__step${n ? '' : ' is-skipped'}` }, h('span', { class: 'path__k', text: 'Evidence' }), n ? `${n} passages` : 'not searched'));
  if (exp.gate2) steps.push(h('span', { class: 'path__step' }, h('span', { class: 'path__k', text: 'Gate 2' }), pill(String(exp.gate2.action))));
  else steps.push(h('span', { class: 'path__step is-skipped' }, h('span', { class: 'path__k', text: 'Gate 2' }), 'skipped'));
  if (finalAction) steps.push(h('span', { class: 'path__step' }, h('span', { class: 'path__k', text: 'Final' }), pill(finalAction)));
  const path = h('div', { class: 'path', role: 'list', 'aria-label': 'Decision path' });
  steps.forEach((step, i) => {
    step.setAttribute('role', 'listitem');
    if (i) path.append(h('span', { class: 'path__arrow', 'aria-hidden': 'true', text: '→' }));
    path.append(step);
  });
  return path;
}

/**
 * @param {any[]} timings - StageTiming[]
 * @param {number} totalMs
 */
function timelineSection(timings, totalMs) {
  const chart = h('div');
  const section = h(
    'section',
    { class: 'why-card' },
    h('div', { class: 'why-card__head' }, h('div', null, h('h4', { text: 'Latency by stage' }),
      h('p', { class: 'why-card__sub', text: `Total ${formatMs(totalMs)}` }))),
    chart,
  );
  renderBarChart(chart, {
    ariaLabel: 'Per-stage latency',
    rows: timings.map((t) => ({
      label: stageShortName(String(t.stage)),
      value: Math.max(0, Number(t.ms) || 0),
      tone: t.status && t.status !== 'ok' ? 'muted' : 'series',
      valueLabel: t.status && t.status !== 'ok' ? `${formatMs(t.ms)} · ${t.status}` : formatMs(t.ms),
    })),
  });
  return section;
}

/**
 * Footnote with models, speculative retrieval, redactions.
 * @param {any} exp
 */
function footnote(exp) {
  const parts = [];
  if (isObject(exp.models) && Object.keys(exp.models).length) {
    parts.push(`Models: ${Object.entries(exp.models).map(([k, v]) => `${k} ${v}`).join(' · ')}`);
  }
  if (isObject(exp.speculative_retrieval)) {
    const s = exp.speculative_retrieval;
    const state = s.used === true ? 'used' : s.discarded === true || s.used === false ? 'discarded (Gate 1 asked first)' : 'on';
    parts.push(`Speculative retrieval: ${state}`);
  }
  if (Number(exp.redactions) > 0) parts.push(`${exp.redactions} personal detail(s) redacted before processing`);
  if (exp.regenerated) parts.push('Regenerated answer');
  if (exp.ablation) parts.push(`Ablation: ${exp.ablation}`);
  return parts.length ? h('p', { class: 'why-foot', text: parts.join('  ·  ') }) : null;
}

/**
 * @param {any[]} calls
 */
function llmCalls(calls) {
  return h(
    'details',
    { class: 'why-calls' },
    h('summary', { text: `Model calls (${calls.length})` }),
    h('div', { class: 'table-wrap' }, h('table', { class: 'data-table' },
      h('thead', null, h('tr', null, ...['Call', 'Model', 'Latency', 'Tokens in/out', 'Cached'].map((t) => h('th', { scope: 'col', text: t })))),
      h('tbody', null, ...calls.filter(isObject).map((c) => h('tr', null,
        h('td', { text: String(c.call_name ?? '') }),
        h('td', { text: String(c.model ?? '') }),
        h('td', { text: formatMs(c.latency_ms) }),
        h('td', { text: `${c.prompt_tokens ?? '—'} / ${c.completion_tokens ?? '—'}` }),
        h('td', { text: c.cached ? 'yes' : 'no' }),
      ))),
    )),
  );
}

/**
 * Build the panel.
 * @param {any} explanation - Explanation model
 * @param {{finalAction: string | null, open: boolean}} options
 * @returns {HTMLDetailsElement | null}
 */
export function explanationPanel(explanation, { finalAction, open }) {
  if (!isObject(explanation)) return null;
  const exp = explanation;
  const body = h('div', { class: 'why__body' }, decisionPath(exp, finalAction));
  if (exp.gate1) body.append(gateCard('Gate 1 · Context Gate', 'Do I know enough about you to search? (SOFT-ASK / PROCEED)', exp.gate1));
  if (Array.isArray(exp.retrieval) && exp.retrieval.length) body.append(evidenceSection(exp.retrieval));
  if (exp.gate2) body.append(gateCard('Gate 2 · Evidence Gate', 'Does the evidence support an answer? (ANSWER / GENERALIZE / ABSTAIN / ESCALATE)', exp.gate2));
  if (isObject(exp.gate3)) body.append(groundingSection(exp.gate3));
  if (isObject(exp.emergency) && Object.keys(exp.emergency).length) body.append(emergencySection(exp.emergency));
  if (isObject(exp.citation_validation)) {
    const cv = exp.citation_validation;
    body.append(h('p', { class: 'why-line' }, icon('book', { size: 14 }),
      `Citations: ${cv.valid ?? 0} valid · ${cv.stripped ?? 0} removed (markers not matching retrieved evidence)`));
  }
  if (Array.isArray(exp.timings) && exp.timings.length) body.append(timelineSection(exp.timings, Number(exp.total_ms) || 0));
  if (Array.isArray(exp.llm_calls) && exp.llm_calls.length) body.append(llmCalls(exp.llm_calls));
  const foot = footnote(exp);
  if (foot) body.append(foot);

  const gateSummary = [exp.gate1 && `Gate 1 ${exp.gate1.action}`, exp.gate2 && `Gate 2 ${exp.gate2.action}`].filter(Boolean).join(' → ');
  return /** @type {HTMLDetailsElement} */ (h(
    'details',
    { class: 'why', open },
    h('summary', null, icon('help', { size: 16 }), h('span', { class: 'why__title', text: 'Why this answer?' }),
      gateSummary ? h('span', { class: 'why__hint', text: gateSummary }) : null,
      h('span', { class: 'why__chev' }, icon('chevronDown', { size: 16 }))),
    body,
  ));
}
