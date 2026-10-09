/**
 * Escape-first Markdown renderer (no third-party code).
 *
 * Every piece of user- or model-supplied text is HTML-escaped BEFORE any
 * markup is added. The renderer then recognises a small, safe Markdown subset
 * and only ever emits tags and attributes it controls:
 *
 *   paragraphs, headings (#..###### mapped to h3-h5), **bold**, *italic*,
 *   ~~strike~~, `inline code`, fenced code blocks, nested ordered/unordered
 *   lists, blockquotes, horizontal rules, simple pipe tables, links
 *   (http/https only, opened in a new tab with rel="noopener noreferrer")
 *   and citation markers [n] -> <button class="cite-chip" data-marker="n">.
 *
 * Because escaping happens first, the working string never contains a raw
 * `<`, `>`, `"` or `'`; injected markup cannot open a tag or break out of an
 * attribute. Link targets are decoded, parsed with the URL constructor and
 * accepted only when the protocol is http: or https:.
 *
 * Pure module: no DOM access, so it is unit-tested under `node --test`.
 */

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const UNESCAPES = { amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'" };
const MAX_DEPTH = 6;
const PLACEHOLDER_RE = /\u0000(\d+)\u0000/g;

const FENCE_RE = /^ {0,3}(`{3,}|~{3,})\s*([\w+#.-]*)[^`]*$/;
const HEADING_RE = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const HR_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const QUOTE_RE = /^ {0,3}>/;
const LIST_ITEM_RE = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const TABLE_SEP_CELL_RE = /^\s*:?-+:?\s*$/;

/**
 * Escape a value for safe use in HTML text and double-quoted attributes.
 * @param {unknown} value
 * @returns {string}
 */
export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ESCAPES[ch]);
}

/** @param {string} value */
function unescapeHtml(value) {
  return value.replace(/&(amp|lt|gt|quot|#39);/g, (_m, name) => UNESCAPES[name]);
}

/**
 * Return a normalised http(s) URL or null when the target is not allowed.
 * @param {string} raw - candidate URL (unescaped)
 * @returns {string | null}
 */
export function safeUrl(raw) {
  const candidate = String(raw ?? '').trim();
  // Reject whitespace / control characters outright (no "java\tscript:" tricks).
  if (!candidate || /[\u0000- \u007f-\u009f\\]/.test(candidate)) return null;
  let parsed;
  try {
    parsed = new URL(candidate);
  } catch {
    return null;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
  return parsed.href;
}

/**
 * Render inline Markdown. `text` is RAW text: it is escaped first.
 * @param {string} text
 * @param {{citations: boolean}} opts
 * @returns {string}
 */
function renderInline(text, opts) {
  const slots = [];
  const stash = (html) => `\u0000${slots.push(html) - 1}\u0000`;
  let s = escapeHtml(text);

  // 1. Code spans: contents are already escaped and are protected from later rules.
  s = s.replace(/``(.+?)``|`([^`\n]+)`/g, (_m, a, b) => stash(`<code>${a ?? b}</code>`));

  // 2. Explicit links [text](url "optional title").
  s = s.replace(
    /\[([^\]\n]+)\]\(\s*([^\s)]+)(?:\s+&quot;[^\n]*?&quot;)?\s*\)/g,
    (_m, label, target) => {
      const href = safeUrl(unescapeHtml(target));
      const inner = applyEmphasis(label);
      if (!href) return stash(inner);
      return stash(
        `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${inner}</a>`,
      );
    },
  );

  // 3. Bare http(s) URLs (stop before any escaped markup character).
  s = s.replace(/\bhttps?:\/\/(?:(?!&lt;|&gt;|&quot;|&#39;)[^\s\u0000])+/g, (match) => {
    const trimmed = match.replace(/(?:[.,;:!?)\]]|&amp;)+$/, '');
    const trailing = match.slice(trimmed.length);
    const href = safeUrl(unescapeHtml(trimmed));
    if (!href) return match;
    return (
      stash(
        `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${trimmed}</a>`,
      ) + trailing
    );
  });

  // 4. Citation markers [1] or [1, 2] -> clickable chips.
  if (opts.citations) {
    s = s.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g, (_m, list) =>
      stash(
        list
          .split(',')
          .map((n) => {
            const marker = String(Number.parseInt(n, 10));
            return (
              `<button type="button" class="cite-chip" data-marker="${marker}" ` +
              `aria-label="Show source ${marker}">${marker}</button>`
            );
          })
          .join(''),
      ),
    );
  }

  // 5. Emphasis.
  s = applyEmphasis(s);

  return restore(s, slots);
}

/**
 * Apply bold / italic / strikethrough to an already-escaped string.
 * @param {string} s
 * @returns {string}
 */
function applyEmphasis(s) {
  // Content classes exclude "<" and ">" so a later rule can never wrap across
  // a tag produced by an earlier one (the working string has no raw "<" or ">").
  return s
    .replace(/\*\*\*(?=[^\s*])([^<>]*?[^\s*<>])\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(?=[^\s*])([^<>]*?[^\s<>])\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^\w])__(?=[^\s_])([^<>]*?[^\s<>])__(?![\w])/g, '$1<strong>$2</strong>')
    .replace(/(^|[^*\w])\*(?=[^\s*])([^*\n<>]*?[^\s*<>])\*(?![*\w])/g, '$1<em>$2</em>')
    .replace(/(^|[^\w])_(?=[^\s_])([^_\n<>]*?[^\s_<>])_(?![\w])/g, '$1<em>$2</em>')
    .replace(/~~(?=\S)([^~\n<>]*?[^\s<>])~~/g, '<del>$1</del>');
}

/**
 * Re-insert stashed HTML fragments (fragments may themselves contain placeholders).
 * @param {string} s
 * @param {string[]} slots
 */
function restore(s, slots) {
  let out = s;
  for (let pass = 0; pass < 4 && out.includes('\u0000'); pass += 1) {
    out = out.replace(PLACEHOLDER_RE, (_m, i) => slots[Number(i)] ?? '');
  }
  return out;
}

/** @param {string} line */
function isBlank(line) {
  return /^\s*$/.test(line);
}

/** @param {string} line */
function indentOf(line) {
  const match = /^(\s*)/.exec(line);
  return match ? match[1].replace(/\t/g, '    ').length : 0;
}

/**
 * Split a pipe-table row into raw cell strings (supports escaped \|).
 * @param {string} line
 * @returns {string[]}
 */
function splitRow(line) {
  let body = line.trim();
  if (body.startsWith('|')) body = body.slice(1);
  if (body.endsWith('|') && !body.endsWith('\\|')) body = body.slice(0, -1);
  const cells = [];
  let current = '';
  for (let i = 0; i < body.length; i += 1) {
    const ch = body[i];
    if (ch === '\\' && body[i + 1] === '|') {
      current += '|';
      i += 1;
    } else if (ch === '|') {
      cells.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  cells.push(current.trim());
  return cells;
}

/**
 * @param {string[]} lines
 * @param {number} i
 */
function isTableStart(lines, i) {
  const header = lines[i];
  const sep = lines[i + 1];
  if (!header || !sep || !header.includes('|') || !sep.includes('-')) return false;
  const sepCells = splitRow(sep);
  if (!sepCells.every((cell) => TABLE_SEP_CELL_RE.test(cell))) return false;
  return splitRow(header).length === sepCells.length;
}

/**
 * Does line `i` open a block other than a paragraph?
 * @param {string[]} lines
 * @param {number} i
 */
function startsBlock(lines, i) {
  const line = lines[i];
  return (
    FENCE_RE.test(line) ||
    HEADING_RE.test(line) ||
    HR_RE.test(line) ||
    QUOTE_RE.test(line) ||
    LIST_ITEM_RE.test(line) ||
    isTableStart(lines, i)
  );
}

/**
 * Remove the common leading indentation from a group of lines.
 * @param {string[]} lines
 */
function dedent(lines) {
  const indents = lines.filter((l) => !isBlank(l)).map(indentOf);
  const min = indents.length ? Math.min(...indents) : 0;
  return lines.map((l) => l.replace(/\t/g, '    ').slice(Math.min(min, indentOf(l))));
}

/**
 * Render a list block (already collected) with nesting.
 * @param {string[]} lines
 * @param {{citations: boolean}} opts
 * @param {number} depth
 */
function renderList(lines, opts, depth) {
  const firstMatch = LIST_ITEM_RE.exec(lines[0]);
  const baseIndent = firstMatch ? indentOf(firstMatch[1]) : 0;
  /** @type {{ordered: boolean, start: number, body: string[]}[]} */
  const items = [];
  for (const line of lines) {
    const match = LIST_ITEM_RE.exec(line);
    if (match && indentOf(match[1]) <= baseIndent + 1) {
      const ordered = /\d/.test(match[2]);
      items.push({ ordered, start: ordered ? Number.parseInt(match[2], 10) : 1, body: [match[3]] });
    } else if (items.length) {
      items[items.length - 1].body.push(line);
    }
  }

  let html = '';
  let k = 0;
  while (k < items.length) {
    const { ordered, start } = items[k];
    const tag = ordered ? 'ol' : 'ul';
    let inner = '';
    while (k < items.length && items[k].ordered === ordered) {
      inner += `<li>${renderListItem(items[k].body, opts, depth)}</li>`;
      k += 1;
    }
    const startAttr = ordered && start !== 1 ? ` start="${start}"` : '';
    html += `<${tag}${startAttr}>${inner}</${tag}>`;
  }
  return html;
}

/**
 * Render a list item: leading text inline, any nested blocks recursively.
 * @param {string[]} body - first line is the item text; the rest is continuation
 * @param {{citations: boolean}} opts
 * @param {number} depth
 */
function renderListItem(body, opts, depth) {
  const rest = dedent(body.slice(1));
  const textLines = [body[0]];
  let j = 0;
  while (j < rest.length && !isBlank(rest[j]) && !startsBlock(rest, j)) {
    textLines.push(rest[j].trim());
    j += 1;
  }
  let html = renderInline(textLines.join('\n'), opts).replace(/\n/g, '<br>');
  const remaining = rest.slice(j);
  if (remaining.some((l) => !isBlank(l))) {
    html +=
      depth < MAX_DEPTH
        ? renderBlocks(remaining, opts, depth + 1)
        : `<p>${renderInline(remaining.join(' '), opts)}</p>`;
  }
  return html;
}

/**
 * Collect the lines that belong to a list starting at `start`.
 * @param {string[]} lines
 * @param {number} start
 * @returns {[string[], number]}
 */
function collectList(lines, start) {
  const block = [];
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) {
      let j = i + 1;
      while (j < lines.length && isBlank(lines[j])) j += 1;
      if (j < lines.length && (LIST_ITEM_RE.test(lines[j]) || indentOf(lines[j]) >= 2)) {
        i = j;
        continue;
      }
      break;
    }
    if (LIST_ITEM_RE.test(line) || indentOf(line) > 0) {
      block.push(line);
    } else if (!startsBlock(lines, i)) {
      block.push(line); // lazy continuation of the previous item
    } else {
      break;
    }
    i += 1;
  }
  return [block, i];
}

/**
 * Render a table starting at line `start`.
 * @param {string[]} lines
 * @param {number} start
 * @param {{citations: boolean}} opts
 * @returns {[string, number]}
 */
function renderTable(lines, start, opts) {
  const header = splitRow(lines[start]);
  const aligns = splitRow(lines[start + 1]).map((cell) => {
    const c = cell.trim();
    if (c.startsWith(':') && c.endsWith(':')) return ' class="al-c"';
    if (c.endsWith(':')) return ' class="al-r"';
    return '';
  });
  let i = start + 2;
  const rows = [];
  while (i < lines.length && !isBlank(lines[i]) && lines[i].includes('|')) {
    rows.push(splitRow(lines[i]));
    i += 1;
  }
  const head = header.map((c, n) => `<th${aligns[n]}>${renderInline(c, opts)}</th>`).join('');
  const body = rows
    .map(
      (row) =>
        `<tr>${header
          .map((_h, n) => `<td${aligns[n]}>${renderInline(row[n] ?? '', opts)}</td>`)
          .join('')}</tr>`,
    )
    .join('');
  return [
    `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`,
    i,
  ];
}

/**
 * Render a sequence of lines as block-level HTML.
 * @param {string[]} lines
 * @param {{citations: boolean}} opts
 * @param {number} depth
 * @returns {string}
 */
function renderBlocks(lines, opts, depth) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) {
      i += 1;
      continue;
    }

    const fence = FENCE_RE.exec(line);
    if (fence) {
      const marker = fence[1];
      const closing = new RegExp(`^ {0,3}${marker[0] === '`' ? '`' : '~'}{${marker.length},}\\s*$`);
      const body = [];
      i += 1;
      while (i < lines.length && !closing.test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      i += 1;
      const lang = fence[2] ? ` class="language-${escapeHtml(fence[2])}"` : '';
      out.push(`<pre><code${lang}>${escapeHtml(body.join('\n'))}</code></pre>`);
      continue;
    }

    const heading = HEADING_RE.exec(line);
    if (heading) {
      const level = Math.min(5, heading[1].length + 2);
      out.push(`<h${level}>${renderInline(heading[2], opts)}</h${level}>`);
      i += 1;
      continue;
    }

    if (HR_RE.test(line)) {
      out.push('<hr>');
      i += 1;
      continue;
    }

    if (QUOTE_RE.test(line)) {
      const inner = [];
      while (i < lines.length && QUOTE_RE.test(lines[i])) {
        inner.push(lines[i].replace(/^ {0,3}> ?/, ''));
        i += 1;
      }
      const content =
        depth < MAX_DEPTH
          ? renderBlocks(inner, opts, depth + 1)
          : `<p>${renderInline(inner.join(' '), opts)}</p>`;
      out.push(`<blockquote>${content}</blockquote>`);
      continue;
    }

    if (isTableStart(lines, i)) {
      const [html, next] = renderTable(lines, i, opts);
      out.push(html);
      i = next;
      continue;
    }

    if (LIST_ITEM_RE.test(line)) {
      const [block, next] = collectList(lines, i);
      out.push(renderList(block, opts, depth));
      i = next;
      continue;
    }

    const para = [line.trim()];
    i += 1;
    while (i < lines.length && !isBlank(lines[i]) && !startsBlock(lines, i)) {
      para.push(lines[i].trim());
      i += 1;
    }
    out.push(`<p>${renderInline(para.join('\n'), opts).replace(/\n/g, '<br>')}</p>`);
  }
  return out.join('');
}

/**
 * Render Markdown to safe HTML.
 * @param {string} source - untrusted Markdown text
 * @param {{citations?: boolean}} [options] - `citations` (default true) turns [n] into chips
 * @returns {string}
 */
export function renderMarkdown(source, options = {}) {
  const opts = { citations: options.citations !== false };
  const text = String(source ?? '')
    .replace(/\u0000/g, '')
    .replace(/\r\n?/g, '\n');
  return renderBlocks(text.split('\n'), opts, 0);
}
