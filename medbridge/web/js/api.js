/**
 * MedBridge HTTP client (same-origin fetch, bearer tokens, SSE streaming).
 *
 * Session routes send `Authorization: Bearer <session_token>` taken from the
 * conversation record kept in the browser store. Errors are raised as
 * `ApiError` carrying the server's message, its `safe_fallback` text, the
 * field-level validation details and the `Retry-After` delay when present.
 */

import { consumeSSE } from './sse.js';

export class ApiError extends Error {
  /**
   * @param {string} message
   * @param {{status?: number, code?: string, safeFallback?: string, aborted?: boolean,
   *          retryAfter?: number | null, details?: {msg: string}[]}} [info]
   */
  constructor(message, info = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = info.status ?? 0;
    this.code = info.code ?? '';
    this.safeFallback =
      info.safeFallback ??
      "We're unable to process your question right now. Please try again, or consult your healthcare provider.";
    this.aborted = Boolean(info.aborted);
    /** Seconds to wait before retrying (HTTP 429 `Retry-After`), or null. */
    this.retryAfter = info.retryAfter ?? null;
    /** Field-level validation messages (HTTP 422), PHI-free. */
    this.details = info.details ?? [];
  }

  /** True when the session token is missing, expired or the session is gone. */
  get sessionUnavailable() {
    return this.sessionGone || this.accessDenied;
  }

  /** The server no longer has the conversation (never existed, or already erased). */
  get sessionGone() {
    return this.status === 404 || this.status === 410;
  }

  /** The session token was rejected (missing, expired or signed with another key). */
  get accessDenied() {
    return this.status === 401 || this.status === 403;
  }
}

/**
 * Parse a `Retry-After` header (delay in seconds or an HTTP date) into whole seconds.
 * @param {string | null | undefined} value
 * @param {number} [now] - epoch ms, for HTTP-date values
 * @returns {number | null}
 */
export function parseRetryAfter(value, now = Date.now()) {
  const raw = String(value ?? '').trim();
  if (!raw) return null;
  if (/^\d+$/.test(raw)) return Number(raw);
  const at = Date.parse(raw);
  if (Number.isNaN(at)) return null;
  return Math.max(0, Math.ceil((at - now) / 1000));
}

/**
 * @typedef {{id: string, token: string | null}} SessionRef
 * @typedef {(message: {type: string, data: any}) => void} StreamHandler
 */

/**
 * Build an ApiError from a failed response.
 * @param {Response} response
 */
async function errorFromResponse(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  const detail = body && typeof body.detail === 'string' ? body.detail : '';
  const rawDetails = Array.isArray(body?.details) ? body.details : Array.isArray(body?.detail) ? body.detail : [];
  const details = rawDetails
    .filter((d) => d && typeof d.msg === 'string' && d.msg.trim())
    .map((d) => ({ msg: String(d.msg).trim() }));
  return new ApiError(body?.message || detail || `Request failed (HTTP ${response.status})`, {
    status: response.status,
    code: body?.error_code || `HTTP_${response.status}`,
    safeFallback: body?.safe_fallback || undefined,
    retryAfter: parseRetryAfter(response.headers?.get?.('Retry-After')),
    details,
  });
}

/**
 * @param {unknown} error
 * @returns {ApiError}
 */
function toApiError(error) {
  if (error instanceof ApiError) return error;
  if (error && /** @type {any} */ (error).name === 'AbortError') {
    return new ApiError('Request cancelled', { aborted: true, code: 'ABORTED' });
  }
  return new ApiError('Unable to reach the MedBridge service.', {
    code: 'NETWORK_ERROR',
    safeFallback:
      'Unable to connect to the MedBridge service. Please check that the server is running and try again.',
  });
}

/**
 * Create the API client.
 * @param {{baseUrl?: string, fetchImpl?: typeof fetch}} [options]
 */
export function createApi({ baseUrl = '', fetchImpl = (...args) => fetch(...args) } = {}) {
  /**
   * @param {string} method
   * @param {string} path
   * @param {{token?: string | null, body?: any, signal?: AbortSignal, accept?: string}} [opts]
   * @returns {Promise<Response>}
   */
  async function send(method, path, { token, body, signal, accept = 'application/json' } = {}) {
    /** @type {Record<string, string>} */
    const headers = { Accept: accept };
    if (token) headers.Authorization = `Bearer ${token}`;
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    let response;
    try {
      response = await fetchImpl(baseUrl + path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal,
        credentials: 'same-origin',
        cache: 'no-store',
      });
    } catch (error) {
      throw toApiError(error);
    }
    if (!response.ok) throw await errorFromResponse(response);
    return response;
  }

  /**
   * @param {string} method
   * @param {string} path
   * @param {{token?: string | null, body?: any, signal?: AbortSignal}} [opts]
   */
  async function json(method, path, opts) {
    const response = await send(method, path, opts);
    if (response.status === 204) return null;
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  /**
   * POST and consume an SSE stream; falls back to a JSON body if the server
   * answered without `text/event-stream` (it is then delivered as `done`).
   * @param {string} path
   * @param {{token?: string | null, body?: any, signal?: AbortSignal, onEvent: StreamHandler}} opts
   */
  async function stream(path, { token, body, signal, onEvent }) {
    const response = await send('POST', path, { token, body, signal, accept: 'text/event-stream' });
    const type = response.headers.get('content-type') || '';
    if (!type.includes('text/event-stream') || !response.body) {
      onEvent({ type: 'done', data: await response.json() });
      return;
    }
    try {
      await consumeSSE(response.body, onEvent);
    } catch (error) {
      throw toApiError(error);
    }
  }

  /** @param {SessionRef} s */
  const base = (s) => `/api/sessions/${encodeURIComponent(s.id)}`;

  return {
    getPublicConfig: () => json('GET', '/api/config/public'),
    createSession: () => json('POST', '/api/sessions'),
    /** @param {SessionRef} s */
    getHistory: (s) => json('GET', `${base(s)}/history`, { token: s.token }),
    /** @param {SessionRef} s */
    deleteSession: (s) => json('DELETE', base(s), { token: s.token }),
    /** @param {SessionRef} s @param {string} message @param {AbortSignal} [signal] */
    sendMessage: (s, message, signal) =>
      json('POST', `${base(s)}/messages`, { token: s.token, body: { message }, signal }),
    /** @param {SessionRef} s @param {string} message @param {AbortSignal} signal @param {StreamHandler} onEvent */
    streamMessage: (s, message, signal, onEvent) =>
      stream(`${base(s)}/messages/stream`, { token: s.token, body: { message }, signal, onEvent }),
    /** @param {SessionRef} s @param {AbortSignal} signal @param {StreamHandler} onEvent */
    streamRegenerate: (s, signal, onEvent) =>
      stream(`${base(s)}/regenerate?stream=true`, { token: s.token, signal, onEvent }),
    /** @param {SessionRef} s @param {AbortSignal} [signal] */
    regenerate: (s, signal) => json('POST', `${base(s)}/regenerate`, { token: s.token, signal }),
    /** @param {SessionRef} s @param {string} messageId @param {1 | -1} rating @param {string} [comment] */
    sendFeedback: (s, messageId, rating, comment) =>
      json('POST', `${base(s)}/feedback`, {
        token: s.token,
        body: comment ? { message_id: messageId, rating, comment } : { message_id: messageId, rating },
      }),
    /** @param {SessionRef} s */
    getContext: (s) => json('GET', `${base(s)}/context`, { token: s.token }),
    /** @param {SessionRef} s @param {{systolic: number, diastolic: number, pulse: number | null, measured_at: string}} reading */
    postBp: (s, reading) => json('POST', `${base(s)}/bp`, { token: s.token, body: reading }),
    /**
     * Clinician summary as Markdown text.
     * @param {SessionRef} s
     * @returns {Promise<string>}
     */
    async getSummaryMarkdown(s) {
      const response = await send('GET', `${base(s)}/summary?format=md`, {
        token: s.token,
        accept: 'text/markdown, application/json;q=0.9',
      });
      const type = response.headers.get('content-type') || '';
      if (type.includes('application/json')) {
        const data = await response.json();
        return String(data?.markdown ?? data?.content ?? data?.summary ?? '');
      }
      return response.text();
    },
    /** @param {string} username @param {string} password */
    adminLogin: (username, password) => json('POST', '/api/admin/login', { body: { username, password } }),
    /** @param {string} token */
    getMetrics: (token) => json('GET', '/api/metrics', { token }),
    /** @param {string} token @param {number} limit */
    getAudit: (token, limit) => json('GET', `/api/admin/audit?limit=${encodeURIComponent(limit)}`, { token }),
  };
}
