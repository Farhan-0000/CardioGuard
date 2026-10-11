/**
 * Tolerant normalisers that turn API payloads into the shapes the UI renders.
 *
 * The backend is the source of truth (docs/finalization_plan.md §3.7); these
 * functions accept the documented keys plus a few close variants so the UI
 * degrades gracefully instead of breaking when an optional field is absent.
 */

import { normalizeAction } from './actions.js';
import { bpCategory, normalizeCategory } from './bp.js';
import { dateStamp, formatDateTime, toDate } from './format.js';

/** Default suggested prompts (used when /api/config/public has none). */
export const FALLBACK_SUGGESTIONS = Object.freeze([
  { title: 'What counts as high', prompt: 'What blood pressure reading counts as high?' },
  { title: 'Lower it naturally', prompt: 'Which diet and exercise changes help lower blood pressure?' },
  { title: 'Medicines', prompt: 'What are the usual first-line medicines for high blood pressure?' },
  { title: 'Targets with diabetes', prompt: 'What blood pressure target is recommended if I have diabetes?' },
  { title: 'Measure at home', prompt: 'How should I measure my blood pressure correctly at home?' },
  { title: 'Warning signs', prompt: 'Which symptoms with high blood pressure mean I should get urgent care?' },
]);

const PHONE_RE = /^\+?\d[\d -]{0,14}$/;

/** @param {unknown} v */
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
/** @param {unknown} v */
const str = (v) => (typeof v === 'string' ? v.trim() : typeof v === 'number' ? String(v) : '');

/**
 * First finite number among `keys` of `obj`.
 * @param {any} obj
 * @param {string[]} keys
 * @returns {number | null}
 */
export function pickNumber(obj, keys) {
  if (!isObject(obj)) return null;
  for (const key of keys) {
    const value = obj[key];
    if (value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value))) {
      return Number(value);
    }
  }
  return null;
}

/**
 * First present value among `keys` of `obj`.
 * @param {any} obj
 * @param {string[]} keys
 */
function pick(obj, keys) {
  if (!isObject(obj)) return undefined;
  for (const key of keys) {
    if (obj[key] !== undefined && obj[key] !== null) return obj[key];
  }
  return undefined;
}

/**
 * @param {unknown} value
 * @param {string} fallback
 */
function phone(value, fallback) {
  const text = str(value);
  return PHONE_RE.test(text) ? text : fallback;
}

/**
 * Normalise GET /api/config/public.
 * @param {any} raw
 */
export function normalizePublicConfig(raw) {
  const cfg = isObject(raw) ? raw : {};
  const emergency = isObject(cfg.emergency) ? cfg.emergency : {};
  const rawSuggestions = cfg.suggestions ?? cfg.suggested_prompts;
  const suggestions = (Array.isArray(rawSuggestions) ? rawSuggestions : [])
    .map((item) => {
      if (typeof item === 'string') return { title: '', prompt: item.trim() };
      if (!isObject(item)) return null;
      const prompt = str(pick(item, ['prompt', 'text', 'query', 'message']));
      return { title: str(pick(item, ['title', 'label', 'heading'])), prompt };
    })
    .filter((s) => s && s.prompt)
    .slice(0, 6);
  const models = isObject(cfg.models) ? cfg.models : {};
  const corpus = isObject(cfg.corpus) ? cfg.corpus : {};
  const corpusItems = Array.isArray(corpus.items) ? corpus.items.filter(isObject) : [];
  const corpusOrgs = [...new Set(corpusItems.map((d) => str(d.org)).filter(Boolean))];
  return {
    appName: str(cfg.app_name) || 'MedBridge',
    tagline:
      str(cfg.tagline || cfg.description) ||
      'Evidence-grounded answers about blood pressure and heart health, from curated clinical guidelines.',
    emergencyNumber: phone(cfg.emergency_number ?? emergency.number, '112'),
    ambulanceNumber: phone(cfg.ambulance_number ?? emergency.ambulance, '108'),
    region:
      str(cfg.emergency_region_label ?? cfg.emergency_region ?? cfg.region ?? emergency.region) ||
      'India',
    localOnly: Boolean(cfg.local_only_mode ?? cfg.local_only ?? false),
    models: /** @type {Record<string, string>} */ (
      Object.fromEntries(Object.entries(models).map(([k, v]) => [k, str(v)]).filter(([, v]) => v))
    ),
    suggestions: suggestions.length ? suggestions : FALLBACK_SUGGESTIONS.map((s) => ({ ...s })),
    corpusCount: pickNumber(corpus, ['documents', 'count']) ?? corpusItems.length,
    corpusOrgs: corpusOrgs.slice(0, 5),
    version: str(cfg.version),
    /** Days of inactivity after which stored conversations are purged (null: not published). */
    retentionDays: (() => {
      const days = pickNumber(cfg, ['data_retention_days', 'retention_days']);
      return days !== null && days > 0 ? Math.round(days) : null;
    })(),
  };
}

/**
 * Normalise one BP reading object.
 * @param {any} item
 * @param {number} index
 */
function normalizeReading(item, index) {
  const systolic = pickNumber(item, ['systolic', 'sbp', 'sys']);
  const diastolic = pickNumber(item, ['diastolic', 'dbp', 'dia']);
  if (systolic === null || diastolic === null) return null;
  const time = toDate(
    pick(item, ['measured_at', 'timestamp', 'recorded_at', 'created_at', 'date', 'time']),
  );
  const category =
    normalizeCategory(pick(item, ['category', 'category_key', 'category_label'])) ??
    bpCategory(systolic, diastolic);
  return {
    systolic,
    diastolic,
    pulse: pickNumber(item, ['pulse', 'heart_rate', 'hr']),
    time,
    category,
    index,
  };
}

/**
 * Normalise GET /api/sessions/{id}/context (snapshot + BP series).
 * @param {any} raw
 */
export function normalizeContext(raw) {
  const payload = isObject(raw) ? raw : {};
  const snapshot = isObject(payload.snapshot) ? payload.snapshot : payload;
  const seriesSource = pick(payload, ['bp_series', 'bp_readings', 'series', 'readings']) ??
    pick(snapshot, ['recent_bp_readings']) ?? [];
  const series = (Array.isArray(seriesSource) ? seriesSource : [])
    .map(normalizeReading)
    .filter((r) => r !== null)
    .sort((a, b) => {
      if (a.time && b.time) return a.time.getTime() - b.time.getTime();
      return a.index - b.index;
    });
  const latestRaw = pick(payload, ['latest_bp', 'latest']);
  const latest = (latestRaw && normalizeReading(latestRaw, series.length)) ||
    series[series.length - 1] || null;

  const meds = (Array.isArray(snapshot.current_medications) ? snapshot.current_medications : [])
    .map((m) =>
      typeof m === 'string'
        ? { name: m, detail: '' }
        : {
            name: str(pick(m, ['drug', 'drug_name', 'name', 'medication'])),
            detail: [str(pick(m, ['dosage', 'dose'])), str(m?.frequency)].filter(Boolean).join(' · '),
          },
    )
    .filter((m) => m.name);
  const symptoms = (Array.isArray(snapshot.symptoms) ? snapshot.symptoms : [])
    .map((s) => (typeof s === 'string' ? s : str(pick(s, ['symptom', 'name']))))
    .filter(Boolean);
  const labs = (Array.isArray(snapshot.lab_results) ? snapshot.lab_results : [])
    .map((l) => ({
      name: str(pick(l, ['lab_type', 'name', 'test'])),
      value: [str(l?.value), str(l?.unit)].filter(Boolean).join(' '),
    }))
    .filter((l) => l.name);
  const demographics = Object.entries(isObject(snapshot.demographics) ? snapshot.demographics : {})
    .map(([key, value]) => [key.replace(/_/g, ' '), Array.isArray(value) ? value.join(', ') : str(value)])
    .filter(([, value]) => value);

  return {
    erased: Boolean(payload.erased),
    latest,
    series,
    medications: meds,
    symptoms,
    labs,
    demographics,
    isEmpty: !latest && !meds.length && !symptoms.length && !labs.length && !demographics.length,
  };
}

/**
 * Normalise a MessageResponse / HistoryMessage into the UI message model.
 * @param {any} raw
 * @param {'user' | 'assistant'} [fallbackRole]
 */
export function normalizeMessage(raw, fallbackRole = 'assistant') {
  const msg = isObject(raw) ? raw : {};
  const role = msg.role === 'user' || msg.role === 'patient' ? 'user' : (msg.role ? 'assistant' : fallbackRole);
  return {
    role,
    content: str(msg.response_text ?? msg.content),
    action: normalizeAction(msg.action),
    citations: Array.isArray(msg.citations) ? msg.citations.filter(isObject) : [],
    timestamp: msg.timestamp ?? msg.created_at ?? null,
    messageId: str(msg.message_id) || null,
    redactionCount: Number(msg.redaction_count ?? msg.explanation?.redactions ?? 0) || 0,
    explanation: isObject(msg.explanation) ? msg.explanation : null,
  };
}

/**
 * Normalise a dict of counts ({"ANSWER": 3} or {"ANSWER": {"count": 3}}).
 * @param {any} value
 * @returns {[string, number][]}
 */
function countEntries(value) {
  if (Array.isArray(value)) {
    return value
      .map((row) => [str(pick(row, ['action', 'decision', 'name', 'key', 'label'])), pickNumber(row, ['count', 'n', 'value'])])
      .filter(([k, v]) => k && v !== null);
  }
  if (!isObject(value)) return [];
  return Object.entries(value)
    .map(([k, v]) => [k, isObject(v) ? pickNumber(v, ['count', 'n', 'value']) : Number(v)])
    .filter(([, v]) => v !== null && Number.isFinite(v));
}

/**
 * Normalise a per-group latency table ({ANSWER: {p50, p95, n}}).
 * @param {any} value
 * @returns {{key: string, p50: number | null, p95: number | null, n: number | null}[]}
 */
function latencyRows(value) {
  const rows = Array.isArray(value)
    ? value.map((row) => [str(pick(row, ['action', 'stage', 'name', 'key'])), row])
    : isObject(value)
      ? Object.entries(value)
      : [];
  return rows
    .map(([key, v]) => ({
      key,
      p50: pickNumber(v, ['p50', 'p50_ms', 'median', 'median_ms']),
      p95: pickNumber(v, ['p95', 'p95_ms']),
      n: pickNumber(v, ['n', 'count']),
    }))
    .filter((r) => r.key && (r.p50 !== null || r.p95 !== null));
}

/** Metric groups shown in the dashboard's safety-mechanisms card. */
const SAFETY_GROUPS = [
  ['routing', 'Routing'],
  ['fallbacks', 'Spec fallbacks'],
  ['citations', 'Citations'],
  ['speculative_retrieval', 'Speculative retrieval'],
  ['grounding_verifier', 'Grounding verifier (Gate 3)'],
  ['redaction', 'PII redaction'],
];

/**
 * Scalar entries of a flat metrics dict, keys humanised.
 * @param {any} value
 * @returns {[string, number][]}
 */
function scalarEntries(value) {
  if (!isObject(value)) return [];
  return Object.entries(value)
    .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
    .map(([k, v]) => [k.replace(/_/g, ' '), v]);
}

/** @param {number} n */
function formatWindow(n) {
  return `≤ ${Math.round(n).toLocaleString()}`;
}

/**
 * Normalise GET /api/metrics.
 * @param {any} raw
 */
export function normalizeMetrics(raw) {
  const m = isObject(raw) ? raw : {};
  const latency = isObject(m.latency) ? m.latency : isObject(m.latency_ms) ? m.latency_ms : {};
  const gates = isObject(m.gates) ? m.gates : isObject(m.gate_counts) ? m.gate_counts : {};
  const feedbackRaw = isObject(m.feedback) ? m.feedback : isObject(m.feedback_stats) ? m.feedback_stats : {};
  const actions = countEntries(pick(m, ['final_actions', 'action_counts', 'actions', 'action_distribution']));
  const sources = isObject(m.gate_sources) ? m.gate_sources : {};
  const windowSize = pickNumber(m, ['window']);
  const up = pickNumber(feedbackRaw, ['up', 'thumbs_up', 'positive', 'likes']) ?? 0;
  const down = pickNumber(feedbackRaw, ['down', 'thumbs_down', 'negative', 'dislikes']) ?? 0;
  const total = pickNumber(feedbackRaw, ['total', 'count', 'n']) ?? up + down;
  const rate =
    pickNumber(feedbackRaw, ['thumbs_up_rate', 'up_rate', 'positive_rate', 'rate']) ??
    (up + down > 0 ? up / (up + down) : null);
  return {
    totalTurns:
      pickNumber(m, ['total_turns', 'turns', 'total', 'total_messages']) ??
      actions.reduce((sum, [, n]) => sum + n, 0),
    p50: pickNumber(m, ['latency_p50_ms', 'p50_ms']) ?? pickNumber(latency, ['p50', 'p50_ms']),
    p95: pickNumber(m, ['latency_p95_ms', 'p95_ms']) ?? pickNumber(latency, ['p95', 'p95_ms']),
    actions,
    gate1: countEntries(pick(m, ['gate1_counts', 'gate_1_counts', 'gate1', 'gate_1']) ?? pick(gates, ['gate1', 'gate_1'])),
    gate2: countEntries(pick(m, ['gate2_counts', 'gate_2_counts', 'gate2', 'gate_2']) ?? pick(gates, ['gate2', 'gate_2'])),
    byAction: latencyRows(pick(m, ['latency_by_action', 'per_action']) ?? pick(latency, ['by_action', 'per_action'])),
    byStage: latencyRows(pick(m, ['stage_latency', 'latency_by_stage', 'per_stage', 'stages']) ?? pick(latency, ['by_stage', 'per_stage'])),
    gate1Sources: countEntries(pick(sources, ['gate1', 'gate_1'])),
    gate2Sources: countEntries(pick(sources, ['gate2', 'gate_2'])),
    safety: SAFETY_GROUPS.map(([key, title]) => ({ title, entries: scalarEntries(m[key]) }))
      .filter((g) => g.entries.length > 0),
    feedback: { up, down, total, rate },
    windowLabel: windowSize !== null ? `Most recent ${formatWindow(windowSize)} turns` : str(pick(m, ['since', 'period'])),
  };
}

/**
 * Normalise GET /api/admin/audit.
 * @param {any} raw
 */
export function normalizeAudit(raw) {
  const rows = Array.isArray(raw)
    ? raw
    : (pick(raw, ['items', 'entries', 'rows', 'logs', 'audit', 'results']) ?? []);
  return (Array.isArray(rows) ? rows : []).filter(isObject).map((row) => ({
    time: toDate(pick(row, ['created_at', 'timestamp', 'time'])),
    action: normalizeAction(pick(row, ['final_action', 'action'])),
    gate1: str(pick(row, ['gate_1_action', 'gate1_action', 'gate1'])),
    gate2: str(pick(row, ['gate_2_action', 'gate2_action', 'gate2'])),
    latency: pickNumber(row, ['latency_ms', 'latency', 'total_ms']),
    request: str(pick(row, ['request_preview', 'request_message', 'request', 'message'])),
    response: str(pick(row, ['response_preview', 'response_text', 'response'])),
    session: str(pick(row, ['session_ref', 'session_id'])).slice(0, 8),
    erased: Boolean(row.erased),
  }));
}

/**
 * Build a Markdown export of a conversation.
 * @param {{title: string, appName: string, messages: ReturnType<typeof normalizeMessage>[]}} input
 * @param {Date} [now]
 * @returns {string}
 */
export function conversationToMarkdown({ title, appName, messages }, now = new Date()) {
  const lines = [
    `# ${title || 'Conversation'}`,
    '',
    `_Exported from ${appName} on ${dateStamp(now)}. General, guideline-based information — ` +
      'not a diagnosis or prescription._',
    '',
  ];
  for (const msg of messages) {
    const who = msg.role === 'user' ? 'You' : appName;
    const meta = [msg.action, formatDateTime(msg.timestamp)].filter(Boolean).join(' · ');
    lines.push(`**${who}**${meta ? ` — ${meta}` : ''}`, '', msg.content || '', '');
    if (msg.citations.length) {
      lines.push('Sources:');
      for (const c of msg.citations) {
        const where = [c.section, c.page ? `p. ${c.page}` : ''].filter(Boolean).join(', ');
        lines.push(`- ${c.marker ?? ''} ${c.title || c.source || 'Guideline'}${where ? ` — ${where}` : ''}`);
      }
      lines.push('');
    }
  }
  return `${lines.join('\n').trimEnd()}\n`;
}
