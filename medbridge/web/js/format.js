/**
 * Small, pure formatting helpers shared by the chat app and the dashboard.
 */

const DAY_MS = 24 * 60 * 60 * 1000;
export const TITLE_MAX_CHARS = 48;

/**
 * Conversation title from the first user message (first 48 characters).
 * @param {string} message
 * @returns {string}
 */
export function titleFromMessage(message) {
  const clean = String(message ?? '').replace(/\s+/g, ' ').trim();
  if (!clean) return 'New conversation';
  if (clean.length <= TITLE_MAX_CHARS) return clean;
  return `${clean.slice(0, TITLE_MAX_CHARS).trimEnd()}…`;
}

/**
 * Neutral conversation label that carries no message text, e.g. "Conversation · 9 Oct 2026".
 * @param {unknown} createdAt - epoch ms / ISO string
 * @returns {string}
 */
export function neutralConversationTitle(createdAt) {
  const date = toDate(createdAt);
  if (!date || date.getTime() <= 0) return 'Conversation';
  const day = date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
  return `Conversation · ${day}`;
}

/**
 * Browser tab title. It never contains message text: browsers keep page titles
 * in their history (and sync it), where "Clear my data" cannot reach.
 * @param {{createdAt?: unknown} | null | undefined} conversation
 * @param {string} appName
 * @returns {string}
 */
export function documentTitle(conversation, appName) {
  return conversation
    ? `${neutralConversationTitle(conversation.createdAt)} · ${appName}`
    : `${appName} — guideline-based heart health assistant`;
}

/**
 * Group conversations (newest first) into Today / Previous 7 days / Older.
 * @template {{updatedAt: number}} T
 * @param {T[]} conversations
 * @param {number} [now]
 * @returns {{label: string, items: T[]}[]}
 */
export function groupConversations(conversations, now = Date.now()) {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const todayStart = startOfToday.getTime();
  const weekStart = todayStart - 7 * DAY_MS;
  const groups = [
    { label: 'Today', items: /** @type {T[]} */ ([]) },
    { label: 'Previous 7 days', items: /** @type {T[]} */ ([]) },
    { label: 'Older', items: /** @type {T[]} */ ([]) },
  ];
  for (const conversation of [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)) {
    if (conversation.updatedAt >= todayStart) groups[0].items.push(conversation);
    else if (conversation.updatedAt >= weekStart) groups[1].items.push(conversation);
    else groups[2].items.push(conversation);
  }
  return groups.filter((g) => g.items.length > 0);
}

/**
 * Human-readable duration.
 * @param {number | null | undefined} ms
 * @returns {string}
 */
export function formatMs(ms) {
  if (ms === null || ms === undefined) return '—';
  const value = Number(ms);
  if (!Number.isFinite(value) || value < 0) return '—';
  if (value < 1000) return `${Math.round(value)} ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)} s`;
  return `${Math.floor(value / 60_000)} min ${Math.round((value % 60_000) / 1000)} s`;
}

/**
 * Parse an ISO string / epoch into a Date (or null).
 * @param {unknown} value
 * @returns {Date | null}
 */
export function toDate(value) {
  if (value === null || value === undefined || value === '') return null;
  const date = value instanceof Date ? value : new Date(/** @type {any} */ (value));
  return Number.isNaN(date.getTime()) ? null : date;
}

/**
 * Short local date-time, e.g. "9 Oct, 14:05".
 * @param {unknown} value
 * @returns {string}
 */
export function formatDateTime(value) {
  const date = toDate(value);
  if (!date) return '';
  return date.toLocaleString(undefined, {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Local time of day, e.g. "14:05".
 * @param {unknown} value
 * @returns {string}
 */
export function formatTime(value) {
  const date = toDate(value);
  return date ? date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) : '';
}

/**
 * Value for an <input type="datetime-local"> in local time.
 * @param {Date} date
 * @returns {string}
 */
export function toLocalInputValue(date) {
  const pad = (n) => String(n).padStart(2, '0');
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}`
  );
}

/**
 * File-name friendly date stamp (YYYY-MM-DD).
 * @param {Date} [date]
 */
export function dateStamp(date = new Date()) {
  return toLocalInputValue(date).slice(0, 10);
}

/**
 * Percentage with no decimals, or an em dash.
 * @param {number | null | undefined} ratio - 0..1
 */
export function formatPercent(ratio) {
  if (ratio === null || ratio === undefined) return '—';
  const value = Number(ratio);
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—';
}

/**
 * Flatten Markdown emphasis/heading/code markers for one-line previews.
 * @param {string} text
 * @returns {string}
 */
export function stripMarkdown(text) {
  return String(text ?? '')
    .replace(/(\*\*|__|~~|`)/g, '')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Thousands-separated integer.
 * @param {number | null | undefined} value
 */
export function formatCount(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.round(n).toLocaleString() : '—';
}
