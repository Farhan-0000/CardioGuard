// Unit tests for the escape-first Markdown renderer (run: node --test medbridge/web/tests).
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { escapeHtml, renderMarkdown, safeUrl } from '../js/markdown.js';

/** Assert the rendered HTML contains no executable or unexpected markup. */
const ALLOWED_TAGS = new Set([
  'p', 'br', 'h3', 'h4', 'h5', 'strong', 'em', 'del', 'code', 'pre', 'ul', 'ol', 'li',
  'blockquote', 'hr', 'div', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'a', 'button',
]);

function assertInert(html) {
  for (const [, name, attrs] of html.matchAll(/<\/?([a-zA-Z0-9]+)([^>]*)>/g)) {
    assert.ok(ALLOWED_TAGS.has(name.toLowerCase()), `unexpected tag <${name}>`);
    assert.doesNotMatch(attrs, /\son[a-z]+\s*=/i, 'no event-handler attributes');
    assert.doesNotMatch(attrs, /\s(style|src|srcdoc|formaction)\s*=/i, 'no dangerous attributes');
    const href = /href="([^"]*)"/.exec(attrs);
    if (href) assert.match(href[1], /^https?:\/\//, 'only http(s) links');
  }
}

test('escapeHtml escapes the five significant characters', () => {
  assert.equal(escapeHtml(`<a href="x">'&'</a>`), '&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;');
  assert.equal(escapeHtml(null), '');
});

test('plain text becomes a paragraph; single newlines become <br>', () => {
  assert.equal(renderMarkdown('Hello\nworld'), '<p>Hello<br>world</p>');
  assert.equal(renderMarkdown('a\n\nb'), '<p>a</p><p>b</p>');
  assert.equal(renderMarkdown(''), '');
});

test('raw HTML is escaped, never interpreted', () => {
  const cases = [
    '<img src=x onerror=alert(1)>',
    '<script>alert(1)</script>',
    '<svg/onload=alert(1)>',
    '<iframe src="javascript:alert(1)"></iframe>',
    '**<img src=x onerror=alert(1)>**',
    '- <b onmouseover="alert(1)">x</b>',
    '> <script>alert(1)</script>',
    '| <img src=x onerror=1> | b |\n|---|---|\n| c | d |',
    '# <script>x</script>',
  ];
  for (const source of cases) {
    const html = renderMarkdown(source);
    assertInert(html);
    assert.match(html, /&lt;/);
  }
});

test('javascript:, data: and other schemes are not linked', () => {
  const sources = [
    '[click](javascript:alert(1))',
    '[click](JaVaScRiPt:alert(1))',
    '[click](  javascript:alert(1))',
    '[click](java\tscript:alert(1))',
    '[click](data:text/html;base64,PHNjcmlwdD4=)',
    '[click](vbscript:msgbox(1))',
    '[click](//evil.example/x)',
    '[click](/relative/path)',
    '[click](file:///etc/passwd)',
  ];
  for (const source of sources) {
    const html = renderMarkdown(source);
    assertInert(html);
    assert.doesNotMatch(html, /<a /, source);
    assert.match(html, /click/);
  }
});

test('attribute break-out attempts inside links stay inert', () => {
  const html = renderMarkdown('[x](https://ok.example/" onmouseover="alert(1))');
  assertInert(html);
  const quoted = renderMarkdown('[x](https://ok.example/a"b)');
  assertInert(quoted);
  assert.doesNotMatch(quoted, /href="[^"]*"[^ >]/);
});

test('http(s) links get target=_blank and rel=noopener noreferrer', () => {
  const html = renderMarkdown('See [AHA](https://www.heart.org/en?a=1&b=2).');
  assert.equal(
    html,
    '<p>See <a href="https://www.heart.org/en?a=1&amp;b=2" target="_blank" ' +
      'rel="noopener noreferrer">AHA</a>.</p>',
  );
  const bare = renderMarkdown('Visit https://www.nice.org.uk/guidance/ng136.');
  assert.match(bare, /<a href="https:\/\/www\.nice\.org\.uk\/guidance\/ng136" target="_blank" rel="noopener noreferrer">/);
  assert.match(bare, /<\/a>\.<\/p>$/);
});

test('safeUrl accepts only http and https', () => {
  assert.equal(safeUrl('https://example.org/x'), 'https://example.org/x');
  assert.equal(safeUrl('http://example.org'), 'http://example.org/');
  assert.equal(safeUrl('javascript:alert(1)'), null);
  assert.equal(safeUrl('mailto:a@b.c'), null);
  assert.equal(safeUrl('https://exa mple.org'), null);
  assert.equal(safeUrl(''), null);
});

test('emphasis, strike and inline code', () => {
  assert.equal(
    renderMarkdown('**bold** and *italic* and _also_ and ~~old~~ and `a<b`'),
    '<p><strong>bold</strong> and <em>italic</em> and <em>also</em> and <del>old</del> and <code>a&lt;b</code></p>',
  );
  assert.equal(renderMarkdown('snake_case_name stays'), '<p>snake_case_name stays</p>');
  assert.equal(renderMarkdown('2 * 3 * 4'), '<p>2 * 3 * 4</p>');
  assert.equal(renderMarkdown('***both***'), '<p><strong><em>both</em></strong></p>');
});

test('code spans protect their content from other rules', () => {
  const html = renderMarkdown('`**not bold** [1] https://x.org`');
  assert.equal(html, '<p><code>**not bold** [1] https://x.org</code></p>');
});

test('fenced code blocks are escaped verbatim', () => {
  const html = renderMarkdown('```html\n<script>alert(1)</script>\n**x**\n```\nafter');
  assert.equal(
    html,
    '<pre><code class="language-html">&lt;script&gt;alert(1)&lt;/script&gt;\n**x**</code></pre><p>after</p>',
  );
  // Unclosed fence (mid-stream) renders to the end without throwing.
  assert.match(renderMarkdown('```\npartial'), /^<pre><code>partial<\/code><\/pre>$/);
});

test('headings map to h3-h5', () => {
  assert.equal(renderMarkdown('# A'), '<h3>A</h3>');
  assert.equal(renderMarkdown('## B'), '<h4>B</h4>');
  assert.equal(renderMarkdown('### C'), '<h5>C</h5>');
  assert.equal(renderMarkdown('###### D'), '<h5>D</h5>');
  assert.equal(renderMarkdown('#hashtag'), '<p>#hashtag</p>');
});

test('unordered and ordered lists, including nesting and inline markup', () => {
  assert.equal(renderMarkdown('- a\n- **b**'), '<ul><li>a</li><li><strong>b</strong></li></ul>');
  assert.equal(renderMarkdown('1. one\n2. two'), '<ol><li>one</li><li>two</li></ol>');
  assert.equal(renderMarkdown('3. three\n4. four'), '<ol start="3"><li>three</li><li>four</li></ol>');
  const nested = renderMarkdown('- parent\n  - child *one*\n    1. deep\n  - child two\n- next');
  assert.equal(
    nested,
    '<ul><li>parent<ul><li>child <em>one</em><ol><li>deep</li></ol></li><li>child two</li></ul></li>' +
      '<li>next</li></ul>',
  );
});

test('list interrupts a paragraph and loose lists keep together', () => {
  assert.equal(
    renderMarkdown('Tips:\n- walk\n\n- sleep\n\nDone.'),
    '<p>Tips:</p><ul><li>walk</li><li>sleep</li></ul><p>Done.</p>',
  );
});

test('code block nested in a list item', () => {
  const html = renderMarkdown('1. Run:\n   ```\n   <b>\n   ```\n2. Next');
  assert.equal(html, '<ol><li>Run:<pre><code>&lt;b&gt;</code></pre></li><li>Next</li></ol>');
});

test('blockquotes support nested markdown', () => {
  assert.equal(
    renderMarkdown('> **Note**\n> - a\n> - b'),
    '<blockquote><p><strong>Note</strong></p><ul><li>a</li><li>b</li></ul></blockquote>',
  );
});

test('deeply nested quotes are depth-limited without throwing', () => {
  const html = renderMarkdown(`${'>'.repeat(200)} deep`);
  assert.match(html, /deep/);
  assertInert(html);
});

test('simple tables with alignment', () => {
  const html = renderMarkdown('| Category | Systolic |\n|:--|--:|\n| Normal | <120 |\n| Stage 1 | 130-139 |');
  assert.equal(
    html,
    '<div class="table-wrap"><table><thead><tr><th>Category</th><th class="al-r">Systolic</th></tr></thead>' +
      '<tbody><tr><td>Normal</td><td class="al-r">&lt;120</td></tr>' +
      '<tr><td>Stage 1</td><td class="al-r">130-139</td></tr></tbody></table></div>',
  );
});

test('horizontal rule', () => {
  assert.equal(renderMarkdown('a\n\n---\n\nb'), '<p>a</p><hr><p>b</p>');
});

test('citation markers become chips; disabled when requested', () => {
  const html = renderMarkdown('Target is <130/80 [1][2] and see [3, 4].');
  assert.equal(
    html,
    '<p>Target is &lt;130/80 ' +
      '<button type="button" class="cite-chip" data-marker="1" aria-label="Show source 1">1</button>' +
      '<button type="button" class="cite-chip" data-marker="2" aria-label="Show source 2">2</button>' +
      ' and see ' +
      '<button type="button" class="cite-chip" data-marker="3" aria-label="Show source 3">3</button>' +
      '<button type="button" class="cite-chip" data-marker="4" aria-label="Show source 4">4</button>.</p>',
  );
  assert.equal(renderMarkdown('see [1]', { citations: false }), '<p>see [1]</p>');
  assert.doesNotMatch(renderMarkdown('[1](https://a.org)'), /cite-chip/);
});

test('NUL characters cannot forge placeholders', () => {
  const html = renderMarkdown('a\u00000\u0000b `x`');
  assert.equal(html, '<p>a0b <code>x</code></p>');
});

test('existing entities are shown literally (double escaped)', () => {
  assert.equal(renderMarkdown('&lt;b&gt;'), '<p>&amp;lt;b&amp;gt;</p>');
});

test('large input renders quickly (no catastrophic backtracking)', () => {
  const source = `${'*'.repeat(5000)} ${'_'.repeat(5000)} ${'['.repeat(3000)} ${'`'.repeat(3000)}`;
  const start = Date.now();
  renderMarkdown(source);
  assert.ok(Date.now() - start < 1500, 'renders in under 1.5 s');
});
