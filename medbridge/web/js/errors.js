/**
 * User-facing text for failed requests.
 *
 * The server's generic `safe_fallback` ("unable to process … call emergency
 * services") is right for outages (5xx) and network failures, but wrong for a
 * request the server deliberately refused: a rate limit (429), a conflict
 * (409, e.g. nothing to regenerate), an oversized or invalid request (413/422).
 * Those carry a fixed, PHI-free, actionable `message`, which is shown instead;
 * a 429 also carries `Retry-After`, shown as a countdown.
 */

/** Wait used when a 429 arrives without a usable `Retry-After` header. */
export const DEFAULT_RETRY_AFTER_SECONDS = 10;

export const SESSION_GONE_TEXT = 'This conversation is no longer available. Please start a new chat.';
export const ACCESS_EXPIRED_TEXT =
  'Access to this conversation has expired in this browser, so it can no longer be opened or changed. Please start a new chat.';

/**
 * @typedef {{status?: number, message?: string, safeFallback?: string, aborted?: boolean,
 *            retryAfter?: number | null, details?: {msg: string}[], code?: string}} ErrorLike
 */

/** @param {ErrorLike | null | undefined} err */
function statusOf(err) {
  return Number(err?.status) || 0;
}

/**
 * True for a request the server refused on purpose (4xx other than auth / missing session).
 * @param {ErrorLike | null | undefined} err
 */
export function isClientError(err) {
  const status = statusOf(err);
  return status >= 400 && status < 500 && ![401, 403, 404, 410].includes(status);
}

/**
 * Seconds to wait before retrying a rate-limited request (0 when not rate limited).
 * @param {ErrorLike | null | undefined} err
 */
export function retryAfterSeconds(err) {
  if (statusOf(err) !== 429) return 0;
  const value = Number(err?.retryAfter);
  return Number.isFinite(value) && value > 0 ? Math.ceil(value) : DEFAULT_RETRY_AFTER_SECONDS;
}

/**
 * Rate-limit notice for the remaining wait.
 * @param {number} seconds
 */
export function rateLimitText(seconds) {
  return seconds > 0
    ? `Too many requests. Please wait ${seconds} s and try again.`
    : 'You can try again now.';
}

/**
 * The server's own message when it is specific (not the client's "Request failed" stub).
 * @param {ErrorLike} err
 * @returns {string}
 */
function serverMessage(err) {
  const message = String(err?.message ?? '').trim();
  if (!message || /^Request failed \(HTTP \d+\)$/.test(message)) return '';
  const details = [...new Set((err.details ?? []).map((d) => String(d.msg).trim()).filter(Boolean))];
  if (statusOf(err) === 422 && details.length) {
    return `${message.replace(/\.$/, '')}: ${details.slice(0, 3).join('; ')}.`;
  }
  return message;
}

/**
 * Text to show for a failed request.
 * @param {ErrorLike | null | undefined} err
 * @param {string} [fallback] - used when the error carries no text at all
 * @returns {string}
 */
export function errorMessage(err, fallback = 'Something went wrong. Please try again.') {
  if (!err) return fallback;
  if (err.aborted) return 'Request cancelled.';
  const status = statusOf(err);
  if (status === 404 || status === 410) return SESSION_GONE_TEXT;
  if (status === 401 || status === 403) return ACCESS_EXPIRED_TEXT;
  if (status === 429) return rateLimitText(retryAfterSeconds(err));
  if (isClientError(err)) return serverMessage(err) || err.safeFallback || fallback;
  // 5xx, network failures and broken streams: the pre-vetted safe fallback.
  return err.safeFallback || fallback;
}

/**
 * True when a failed turn may still be running (and saving) on the server:
 * the request was cut off (stop, lost connection, stream ended early) rather
 * than answered with an HTTP error status or a server-sent `error` event.
 * @param {ErrorLike | null | undefined} err
 */
export function mayStillRunOnServer(err) {
  if (!err) return false;
  if (err.aborted) return true;
  return statusOf(err) === 0 && err.code !== 'STREAM_ERROR';
}

/**
 * What a failed `DELETE /api/sessions/{id}` means for the conversation kept in
 * this browser:
 * - `gone`: the server no longer has it (404/410) — safe to drop locally;
 * - `denied`: the access token was rejected (401/403) — nothing was erased, so
 *   the local record and its key must be kept and the user told so;
 * - `failed`: anything else (5xx, network, 429) — nothing was erased; retry later.
 * @param {ErrorLike | null | undefined} err
 * @returns {'gone' | 'denied' | 'failed'}
 */
export function classifyEraseError(err) {
  const status = statusOf(err);
  if (status === 404 || status === 410) return 'gone';
  if (status === 401 || status === 403) return 'denied';
  return 'failed';
}

/**
 * Call `onTick(remaining)` now and once per second until it reaches 0.
 * @param {number} seconds
 * @param {(remaining: number) => void} onTick
 * @param {{setIntervalImpl?: typeof setInterval, clearIntervalImpl?: typeof clearInterval}} [timers]
 * @returns {() => void} cancel
 */
export function startCountdown(seconds, onTick, timers = {}) {
  const setI = timers.setIntervalImpl ?? setInterval;
  const clearI = timers.clearIntervalImpl ?? clearInterval;
  let remaining = Math.max(0, Math.ceil(seconds));
  onTick(remaining);
  if (remaining === 0) return () => {};
  const handle = setI(() => {
    remaining -= 1;
    onTick(remaining);
    if (remaining <= 0) clearI(handle);
  }, 1000);
  return () => clearI(handle);
}
