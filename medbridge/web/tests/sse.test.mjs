// Unit tests for the POST-SSE parser.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { SSEParser, consumeSSE, tokenText } from '../js/sse.js';

function collect() {
  const events = [];
  return { events, parser: new SSEParser((e) => events.push(e)) };
}

test('parses event + JSON data blocks', () => {
  const { events, parser } = collect();
  parser.push('event: stage\ndata: {"stage":"emergency","status":"start"}\n\n');
  parser.push('event: token\ndata: {"text":"Hi"}\n\n');
  assert.deepEqual(events, [
    { type: 'stage', data: { stage: 'emergency', status: 'start' } },
    { type: 'token', data: { text: 'Hi' } },
  ]);
});

test('handles chunks split anywhere, CRLF split across chunks and comments', () => {
  const { events, parser } = collect();
  const wire = ': keep-alive\r\nevent: done\r\ndata: {"action":"ANSWER",\r\ndata: "n":1}\r\n\r\n';
  for (const ch of wire) parser.push(ch);
  parser.end();
  assert.deepEqual(events, [{ type: 'done', data: { action: 'ANSWER', n: 1 } }]);
});

test('lone CR line endings and default message type', () => {
  const { events, parser } = collect();
  parser.push('data: plain text\r\rdata: {"a":1}\r\r');
  parser.end();
  assert.deepEqual(events, [
    { type: 'message', data: 'plain text' },
    { type: 'message', data: { a: 1 } },
  ]);
});

test('end() flushes a final event without trailing blank line', () => {
  const { events, parser } = collect();
  parser.push('event: done\ndata: {"ok":true}');
  parser.end();
  assert.deepEqual(events, [{ type: 'done', data: { ok: true } }]);
});

test('an event without data is not dispatched', () => {
  const { events, parser } = collect();
  parser.push('event: ping\n\n');
  parser.end();
  assert.equal(events.length, 0);
});

test('consumeSSE reads a ReadableStream of bytes (multi-byte chars split)', async () => {
  const bytes = new TextEncoder().encode('event: token\ndata: {"text":"héllo — ✓"}\n\n');
  const stream = new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += 3) controller.enqueue(bytes.slice(i, i + 3));
      controller.close();
    },
  });
  const events = [];
  await consumeSSE(stream, (e) => events.push(e));
  assert.deepEqual(events, [{ type: 'token', data: { text: 'héllo — ✓' } }]);
});

test('tokenText accepts the known payload keys', () => {
  assert.equal(tokenText({ text: 'a' }), 'a');
  assert.equal(tokenText({ token: 'b' }), 'b');
  assert.equal(tokenText({ delta: 'c' }), 'c');
  assert.equal(tokenText({ content: 'd' }), 'd');
  assert.equal(tokenText('e'), 'e');
  assert.equal(tokenText({ other: 1 }), '');
  assert.equal(tokenText(null), '');
});
