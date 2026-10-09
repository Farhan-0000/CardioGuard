/**
 * The five MedBridge routing actions and the pipeline stage labels.
 *
 * Colours live in CSS (`.badge--answer` ...); this module only carries the
 * wording so the chat, the legacy Streamlit UI and the dashboard agree.
 */

/** @typedef {'ANSWER' | 'GENERALIZE' | 'SOFT-ASK' | 'ABSTAIN' | 'ESCALATE'} Action */

export const ACTIONS = /** @type {const} */ ([
  'ANSWER',
  'GENERALIZE',
  'SOFT-ASK',
  'ABSTAIN',
  'ESCALATE',
]);

/** @type {Record<Action, {tone: string, label: string, tooltip: string}>} */
export const ACTION_META = {
  ANSWER: {
    tone: 'answer',
    label: 'Answer',
    tooltip: 'Answered from guideline evidence that matches your situation',
  },
  GENERALIZE: {
    tone: 'generalize',
    label: 'General info',
    tooltip: 'General guideline information — not personalised',
  },
  'SOFT-ASK': {
    tone: 'softask',
    label: 'Needs detail',
    tooltip: 'I need a bit more information to answer safely',
  },
  ABSTAIN: {
    tone: 'abstain',
    label: 'Outside scope',
    tooltip: 'Outside what my guidelines can safely answer',
  },
  ESCALATE: {
    tone: 'escalate',
    label: 'Seek care now',
    tooltip: 'Possible emergency — seek care now',
  },
};

/**
 * Normalise an action string ("soft_ask", "SOFT-ASK", ...) or return null.
 * @param {unknown} value
 * @returns {Action | null}
 */
export function normalizeAction(value) {
  const key = String(value ?? '')
    .toUpperCase()
    .replace(/_/g, '-')
    .trim();
  return /** @type {Action | null} */ (ACTIONS.includes(/** @type {any} */ (key)) ? key : null);
}

/** Patient-facing stage labels (mirror of medbridge.core.pipeline_types.STAGE_LABELS). */
export const STAGE_LABELS = Object.freeze({
  privacy: 'Protecting your privacy…',
  emergency: 'Checking for emergencies…',
  load_state: 'Loading your conversation…',
  small_talk: 'Reading your message…',
  extraction: 'Understanding your context…',
  context_gate: 'Checking what I know about you…',
  retrieval: 'Searching guidelines…',
  rerank: 'Ranking the evidence…',
  evidence_gate: 'Verifying evidence…',
  generation: 'Writing answer…',
  verification: 'Double-checking claims…',
  persistence: 'Saving securely…',
});

/** Short technical names used in the latency timeline. */
export const STAGE_SHORT = Object.freeze({
  privacy: 'Privacy',
  emergency: 'Emergency check',
  load_state: 'Load state',
  small_talk: 'Small talk',
  extraction: 'Extraction',
  context_gate: 'Gate 1 · Context',
  retrieval: 'Retrieval',
  rerank: 'Rerank',
  evidence_gate: 'Gate 2 · Evidence',
  generation: 'Generation',
  verification: 'Gate 3 · Grounding',
  persistence: 'Persistence',
});

/**
 * @param {string} stage
 * @returns {string}
 */
export function stageShortName(stage) {
  return /** @type {Record<string, string>} */ (STAGE_SHORT)[stage] ?? stage.replace(/_/g, ' ');
}
