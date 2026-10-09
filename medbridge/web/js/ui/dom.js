/**
 * Minimal DOM helpers. Text is always set with textContent; the only HTML
 * strings ever assigned come from the escape-first Markdown renderer.
 */

/**
 * @typedef {Record<string, any>} Attrs
 * Keys: `class`, `text`, `dataset` (object), `style` (object), `on` (event map),
 * boolean attributes (true/false), any other attribute as string/number.
 */

/**
 * Create an element.
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tag
 * @param {Attrs | null} [attrs]
 * @param {...(Node | string | null | undefined | false)} children
 * @returns {HTMLElementTagNameMap[K]}
 */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  applyAttrs(el, attrs);
  append(el, ...children);
  return el;
}

/**
 * @param {Element} el
 * @param {Attrs | null | undefined} attrs
 */
export function applyAttrs(el, attrs) {
  if (!attrs) return;
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.setAttribute('class', String(value));
    else if (key === 'text') el.textContent = String(value);
    else if (key === 'dataset') Object.assign(/** @type {HTMLElement} */ (el).dataset, value);
    else if (key === 'style') Object.assign(/** @type {HTMLElement} */ (el).style, value);
    else if (key === 'on') {
      for (const [type, handler] of Object.entries(value)) el.addEventListener(type, handler);
    } else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, String(value));
  }
}

/**
 * Append children, skipping empty values; strings become text nodes.
 * @param {Element} el
 * @param {...(Node | string | null | undefined | false)} children
 */
export function append(el, ...children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    el.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
}

/** @param {Element} el */
export function clear(el) {
  while (el.firstChild) el.firstChild.remove();
}

/**
 * Query a required element by id.
 * @param {string} id
 * @returns {HTMLElement}
 */
export function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el;
}

/** True when the user asked the OS for reduced motion. */
export function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Trigger a client-side download of text content.
 * @param {string} filename
 * @param {string} content
 * @param {string} [type]
 */
export function downloadText(filename, content, type = 'text/markdown;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = h('a', { href: url, download: filename, hidden: true });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Copy text to the clipboard (Clipboard API with a textarea fallback).
 * @param {string} text
 * @returns {Promise<boolean>}
 */
export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall back below
  }
  const area = h('textarea', { 'aria-hidden': 'true', readonly: true, class: 'clipboard-proxy' });
  area.value = text;
  document.body.append(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  area.remove();
  return ok;
}

/** Elements that can receive keyboard focus inside a container. */
export const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), ' +
  'textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

/**
 * Keep Tab / Shift+Tab inside `container`. Returns a function that removes the trap.
 * @param {HTMLElement} container
 * @returns {() => void}
 */
export function trapFocus(container) {
  /** @param {KeyboardEvent} event */
  const onKey = (event) => {
    if (event.key !== 'Tab') return;
    const items = [...container.querySelectorAll(FOCUSABLE)].filter(
      (el) => /** @type {HTMLElement} */ (el).offsetParent !== null || el === document.activeElement,
    );
    if (!items.length) {
      event.preventDefault();
      return;
    }
    const first = /** @type {HTMLElement} */ (items[0]);
    const last = /** @type {HTMLElement} */ (items[items.length - 1]);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };
  container.addEventListener('keydown', onKey);
  return () => container.removeEventListener('keydown', onKey);
}
