/**
 * Blood-pressure helpers: ACC/AHA categories and manual-entry validation.
 *
 * The server sends a category per reading; `bpCategory` is only the fallback
 * used when it does not. Thresholds follow the ACC/AHA adult categories:
 * normal <120 and <80, elevated 120-129 and <80, stage 1 130-139 or 80-89,
 * stage 2 >=140 or >=90, severe / crisis range >=180 or >=120 (same rules as
 * medbridge.features.bp on the server).
 */

/** @typedef {'normal' | 'elevated' | 'stage1' | 'stage2' | 'crisis'} BpCategoryKey */

/** Ordered categories with their systolic band (used to shade the trend chart). */
export const BP_CATEGORIES = /** @type {const} */ ([
  { key: 'normal', label: 'Normal', sysFrom: 0, sysTo: 120 },
  { key: 'elevated', label: 'Elevated', sysFrom: 120, sysTo: 130 },
  { key: 'stage1', label: 'Stage 1', sysFrom: 130, sysTo: 140 },
  { key: 'stage2', label: 'Stage 2', sysFrom: 140, sysTo: 180 },
  { key: 'crisis', label: 'Crisis', sysFrom: 180, sysTo: 400 },
]);

/** Limits mirrored from the server-side validation of POST /bp. */
export const BP_LIMITS = Object.freeze({
  systolic: { min: 60, max: 300 },
  diastolic: { min: 30, max: 200 },
  pulse: { min: 20, max: 250 },
});

/**
 * @param {BpCategoryKey} key
 */
export function categoryLabel(key) {
  return BP_CATEGORIES.find((c) => c.key === key)?.label ?? 'Unknown';
}

/**
 * ACC/AHA category for one reading.
 * @param {number} systolic
 * @param {number} diastolic
 * @returns {BpCategoryKey}
 */
export function bpCategory(systolic, diastolic) {
  if (systolic >= 180 || diastolic >= 120) return 'crisis';
  if (systolic >= 140 || diastolic >= 90) return 'stage2';
  if (systolic >= 130 || diastolic >= 80) return 'stage1';
  if (systolic >= 120) return 'elevated';
  return 'normal';
}

/**
 * Map a server category string ("Stage 1 hypertension", "STAGE_2",
 * "hypertensive_crisis", ...) to a category key.
 * @param {unknown} value
 * @returns {BpCategoryKey | null}
 */
export function normalizeCategory(value) {
  const text = String(value ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
  if (!text) return null;
  if (/crisis|severe|emergency|urgency/.test(text)) return 'crisis';
  if (/stage ?2|stage ?ii\b/.test(text)) return 'stage2';
  if (/stage ?1|stage ?i\b/.test(text)) return 'stage1';
  if (/elevated/.test(text)) return 'elevated';
  if (/normal|optimal/.test(text)) return 'normal';
  return null;
}

/**
 * Validate the manual BP form. Returns typed values or per-field errors.
 * @param {{systolic: string, diastolic: string, pulse?: string, measuredAt?: string}} form
 * @param {number} [now]
 * @returns {{ok: boolean, errors: Record<string, string>,
 *            value: {systolic: number, diastolic: number, pulse: number | null, measured_at: string}}}
 */
export function validateBpInput(form, now = Date.now()) {
  /** @type {Record<string, string>} */
  const errors = {};
  const readInt = (raw, field, label, required) => {
    const text = String(raw ?? '').trim();
    if (!text) {
      if (required) errors[field] = `Enter your ${label} pressure.`;
      return null;
    }
    if (!/^\d{1,3}$/.test(text)) {
      errors[field] = 'Use whole numbers only.';
      return null;
    }
    const value = Number.parseInt(text, 10);
    const { min, max } = BP_LIMITS[/** @type {'systolic'|'diastolic'|'pulse'} */ (field)];
    if (value < min || value > max) {
      errors[field] = `Must be between ${min} and ${max}.`;
      return null;
    }
    return value;
  };

  const systolic = readInt(form.systolic, 'systolic', 'systolic (top)', true);
  const diastolic = readInt(form.diastolic, 'diastolic', 'diastolic (bottom)', true);
  const pulse = readInt(form.pulse, 'pulse', 'pulse', false);
  if (systolic !== null && diastolic !== null && systolic <= diastolic) {
    errors.diastolic = 'The bottom number must be lower than the top number.';
  }

  let measuredAt = new Date(now);
  if (form.measuredAt) {
    const parsed = new Date(form.measuredAt);
    if (Number.isNaN(parsed.getTime())) errors.measuredAt = 'Enter a valid date and time.';
    else if (parsed.getTime() > now + 5 * 60 * 1000) errors.measuredAt = 'The time cannot be in the future.';
    else measuredAt = parsed;
  }

  return {
    ok: Object.keys(errors).length === 0,
    errors,
    value: {
      systolic: systolic ?? 0,
      diastolic: diastolic ?? 0,
      pulse,
      measured_at: measuredAt.toISOString(),
    },
  };
}
