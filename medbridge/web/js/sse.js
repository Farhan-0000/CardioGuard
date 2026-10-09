/**
 * Server-Sent Events over a POST response (fetch + ReadableStream).
 *
 * `EventSource` only supports GET, so the chat stream is consumed by reading
 * the fetch body and parsing the SSE wire format ourselves:
 *
 *   event: <type>\n
 *   data: <json>\n
 *   \n
 *
 * Multi-line `data:` fields are joined with "\n", comment lines (":") are
 * ignored, and CR, LF and CRLF line endings are all accepted - also when a
 * CRLF pair is split across two network chunks.
 */

/**
 * @typedef {{type: string, data: any}} SSEMessage
 */

export class SSEParser {
  /** @param {(message: SSEMessage) => void} onMessage */
  constructor(onMessage) {
    this.onMessage = onMessage;
    this.buffer = '';
    this.eventType = '';
    /** @type {string[]} */
    this.dataLines = [];
  }

  /**
   * Feed a decoded text chunk.
   * @param {string} chunk
   */
  push(chunk) {
    this.buffer += chunk;
    for (;;) {
      const match = /\r\n|\r|\n/.exec(this.buffer);
      if (!match) return;
      // A trailing lone CR may be the first half of a CRLF split across chunks.
      if (match[0] === '\r' && match.index === this.buffer.length - 1) return;
      const line = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      this.processLine(line);
    }
  }

  /** Flush any buffered data at end of stream. */
  end() {
    if (this.buffer) {
      const rest = this.buffer.replace(/\r$/, '');
      this.buffer = '';
      if (rest) this.processLine(rest);
    }
    this.dispatch();
  }

  /** @param {string} line */
  processLine(line) {
    if (line === '') {
      this.dispatch();
      return;
    }
    if (line.startsWith(':')) return;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') this.eventType = value;
    else if (field === 'data') this.dataLines.push(value);
  }

  dispatch() {
    if (this.dataLines.length === 0) {
      this.eventType = '';
      return;
    }
    const type = this.eventType || 'message';
    const raw = this.dataLines.join('\n');
    this.eventType = '';
    this.dataLines = [];
    let data = raw;
    try {
      data = JSON.parse(raw);
    } catch {
      // Non-JSON payloads are delivered as plain strings.
    }
    this.onMessage({ type, data });
  }
}

/**
 * Read an SSE body to completion, invoking `onMessage` per event.
 * @param {ReadableStream<Uint8Array>} body
 * @param {(message: SSEMessage) => void} onMessage
 * @returns {Promise<void>}
 */
export async function consumeSSE(body, onMessage) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser(onMessage);
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.end();
  } finally {
    reader.releaseLock();
  }
}

/**
 * Extract the text delta from a `token` event payload.
 * Accepts `{text}`, `{token}`, `{delta}` or `{content}` (or a bare string).
 * @param {any} data
 * @returns {string}
 */
export function tokenText(data) {
  if (typeof data === 'string') return data;
  if (!data || typeof data !== 'object') return '';
  for (const key of ['text', 'token', 'delta', 'content']) {
    if (typeof data[key] === 'string') return data[key];
  }
  return '';
}
