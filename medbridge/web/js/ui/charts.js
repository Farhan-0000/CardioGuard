/**
 * Hand-written SVG charts (no chart library).
 *
 * Charts are rendered at the container's real pixel width (re-rendered by a
 * ResizeObserver) so text never scales. Mark specs: bars <= 16px thick with a
 * 4px rounded data end and a square baseline; values labelled at the bar tip
 * in text colour; a hover/focus tooltip per row; a data-table view lives
 * alongside every chart for screen readers and print.
 */

import { h } from './dom.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Create an SVG element with attributes.
 * @param {string} tag
 * @param {Record<string, string | number>} [attrs]
 * @returns {SVGElement}
 */
export function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  return el;
}

/**
 * Path for a horizontal bar with a rounded right end and a square left end.
 * @param {number} x
 * @param {number} y
 * @param {number} width
 * @param {number} height
 * @param {number} [radius]
 */
export function barPath(x, y, width, height, radius = 4) {
  const w = Math.max(0, width);
  const r = Math.min(radius, w, height / 2);
  return (
    `M${x} ${y}H${x + w - r}` +
    `A${r} ${r} 0 0 1 ${x + w} ${y + r}V${y + height - r}` +
    `A${r} ${r} 0 0 1 ${x + w - r} ${y + height}H${x}Z`
  );
}

/**
 * Re-render `draw(width)` into `container` whenever its width changes.
 * @param {HTMLElement} container
 * @param {(width: number) => Element} draw
 */
export function responsive(container, draw) {
  let lastWidth = -1;
  const render = () => {
    const width = Math.max(220, Math.floor(container.clientWidth || 320));
    if (width === lastWidth) return;
    lastWidth = width;
    container.replaceChildren(draw(width));
  };
  render();
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(() => {
      if (!container.isConnected) {
        observer.disconnect();
        return;
      }
      render();
    });
    observer.observe(container);
  }
}

/**
 * Shared floating tooltip bound to a chart container.
 * @param {HTMLElement} host - positioned container
 */
export function createTooltip(host) {
  const tip = h('div', { class: 'chart-tip', role: 'presentation', hidden: true });
  host.append(tip);
  return {
    /** @param {string} text @param {number} x @param {number} y */
    show(text, x, y) {
      tip.textContent = text;
      tip.hidden = false;
      const maxLeft = host.clientWidth - tip.offsetWidth - 4;
      tip.style.left = `${Math.max(4, Math.min(maxLeft, x - tip.offsetWidth / 2))}px`;
      tip.style.top = `${Math.max(0, y - tip.offsetHeight - 10)}px`;
    },
    hide() {
      tip.hidden = true;
    },
  };
}

/**
 * @typedef {{label: string, value: number, track?: number | null, tone?: string,
 *            valueLabel: string, tooltip?: string}} BarRow
 *   `track` draws a lighter bar behind the value bar (e.g. p95 behind p50).
 */

/**
 * Horizontal bar chart.
 * @param {HTMLElement} container
 * @param {{rows: BarRow[], ariaLabel: string, labelWidth?: number}} options
 */
export function renderBarChart(container, { rows, ariaLabel, labelWidth }) {
  container.classList.add('chart');
  const slot = h('div');
  const host = h('div', { class: 'chart__plot' }, slot);
  container.replaceChildren(host);
  if (!rows.length) {
    host.replaceChildren(h('p', { class: 'chart__empty', text: 'No data yet.' }));
    return;
  }
  const tooltip = createTooltip(host);
  const max = Math.max(...rows.map((r) => Math.max(r.value, r.track ?? 0)), 1);
  const rowH = 30;
  const barH = 14;

  responsive(slot, (width) => {
    const longest = Math.max(...rows.map((r) => r.label.length));
    const labelW = labelWidth ?? Math.min(Math.round(width * 0.38), 16 + longest * 7);
    const valueW = Math.min(132, Math.max(56, 8 + Math.max(...rows.map((r) => r.valueLabel.length)) * 6.6));
    const plotW = Math.max(40, width - labelW - valueW - 8);
    const height = rows.length * rowH + 4;
    const svg = svgEl('svg', {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: 'img',
      'aria-label': ariaLabel,
      class: 'chart__svg',
    });
    svg.append(svgEl('line', { x1: labelW, x2: labelW, y1: 0, y2: height, class: 'chart__axis' }));
    rows.forEach((row, i) => {
      const y = i * rowH + (rowH - barH) / 2 + 2;
      const text = row.tooltip ?? `${row.label}: ${row.valueLabel}`;
      const g = svgEl('g', {
        class: `chart__row tone-${row.tone ?? 'series'}`,
        tabindex: 0,
        role: 'img',
        'aria-label': text,
      });
      const label = svgEl('text', { x: labelW - 10, y: y + barH / 2, class: 'chart__label', 'text-anchor': 'end', 'dominant-baseline': 'central' });
      label.textContent = row.label;
      g.append(label);
      if (row.track) {
        g.append(svgEl('path', { d: barPath(labelW, y, (row.track / max) * plotW, barH), class: 'chart__track' }));
      }
      const valueWidth = Math.max(row.value > 0 ? 2 : 0, (row.value / max) * plotW);
      g.append(svgEl('path', { d: barPath(labelW, y, valueWidth, barH), class: 'chart__bar' }));
      const tipX = labelW + Math.max(valueWidth, ((row.track ?? 0) / max) * plotW) + 8;
      const value = svgEl('text', { x: tipX, y: y + barH / 2, class: 'chart__value', 'dominant-baseline': 'central' });
      value.textContent = row.valueLabel;
      g.append(value);
      g.append(svgEl('rect', { x: 0, y: i * rowH, width, height: rowH, class: 'chart__hit' }));
      const show = () => tooltip.show(text, labelW + valueWidth / 2, y);
      g.addEventListener('pointerenter', show);
      g.addEventListener('focus', show);
      g.addEventListener('pointerleave', () => tooltip.hide());
      g.addEventListener('blur', () => tooltip.hide());
      svg.append(g);
    });
    return svg;
  });
}

/**
 * Collapsible data table that mirrors a chart.
 * @param {string} caption
 * @param {string[]} headers
 * @param {(string | number)[][]} rows
 */
export function dataTable(caption, headers, rows) {
  return h(
    'details',
    { class: 'table-view' },
    h('summary', { text: 'Show data table' }),
    h(
      'div',
      { class: 'table-wrap' },
      h(
        'table',
        { class: 'data-table' },
        h('caption', { class: 'visually-hidden', text: caption }),
        h('thead', null, h('tr', null, ...headers.map((t) => h('th', { scope: 'col', text: t })))),
        h('tbody', null, ...rows.map((r) => h('tr', null, ...r.map((c) => h('td', { text: String(c) }))))),
      ),
    ),
  );
}
