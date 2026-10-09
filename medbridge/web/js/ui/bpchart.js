/**
 * Blood-pressure trend chart (hand-written SVG).
 *
 * Systolic and diastolic lines over time on one mmHg axis, with the ACC/AHA
 * systolic category bands shaded behind them (normal / elevated / stage 1 /
 * stage 2 / crisis). Every reading is a focusable marker with a tooltip; the
 * SVG carries <title>/<desc> and a legend explains lines and bands.
 */

import { BP_CATEGORIES, categoryLabel } from '../bp.js';
import { formatDateTime } from '../format.js';
import { createTooltip, responsive, svgEl } from './charts.js';
import { h } from './dom.js';

let chartCount = 0;

/**
 * @typedef {{systolic: number, diastolic: number, pulse: number | null,
 *            time: Date | null, category: string}} Reading
 */

/**
 * @param {Reading} r
 */
function describe(r) {
  const parts = [`${r.systolic}/${r.diastolic} mmHg`, categoryLabel(/** @type {any} */ (r.category))];
  if (r.pulse) parts.push(`pulse ${r.pulse}`);
  const when = formatDateTime(r.time);
  if (when) parts.push(when);
  return parts.join(' · ');
}

/**
 * Render the chart (plus legend) into `container`.
 * @param {HTMLElement} container
 * @param {Reading[]} series - sorted oldest first
 */
export function renderBpChart(container, series) {
  chartCount += 1;
  const titleId = `bp-title-${chartCount}`;
  const descId = `bp-desc-${chartCount}`;
  const slot = h('div');
  const host = h('div', { class: 'chart__plot bp-chart' }, slot);
  const tooltip = createTooltip(host);
  const legend = h(
    'div',
    { class: 'bp-legend', 'aria-hidden': 'true' },
    h('span', { class: 'bp-legend__line bp-legend__line--sys' }, 'Systolic'),
    h('span', { class: 'bp-legend__line bp-legend__line--dia' }, 'Diastolic'),
    ...BP_CATEGORIES.map((c) => h('span', { class: `bp-legend__band band-${c.key}` }, c.label)),
  );
  container.replaceChildren(host, legend);

  const values = series.flatMap((r) => [r.systolic, r.diastolic]);
  const yMin = Math.max(20, Math.floor((Math.min(70, ...values) - 4) / 20) * 20);
  const yMax = Math.ceil((Math.max(185, ...values) + 4) / 20) * 20;
  const times = series.map((r) => (r.time ? r.time.getTime() : null));
  const useTime = times.every((t) => t !== null) && new Set(times).size === series.length && series.length > 1;

  responsive(slot, (width) => {
    const height = 210;
    const pad = { top: 22, right: 10, bottom: 26, left: 38 };
    const inset = 12;
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const y = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
    const tMin = useTime ? Math.min(.../** @type {number[]} */ (times)) : 0;
    const tMax = useTime ? Math.max(.../** @type {number[]} */ (times)) : Math.max(1, series.length - 1);
    /** @param {number} i */
    const x = (i) => {
      if (series.length === 1) return pad.left + plotW / 2;
      const t = useTime ? /** @type {number} */ (times[i]) : i;
      return pad.left + inset + ((t - tMin) / (tMax - tMin)) * (plotW - 2 * inset);
    };

    const svg = svgEl('svg', {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: 'img',
      'aria-labelledby': `${titleId} ${descId}`,
      class: 'chart__svg',
    });
    const title = svgEl('title', { id: titleId });
    title.textContent = 'Blood pressure trend';
    const latest = series[series.length - 1];
    const desc = svgEl('desc', { id: descId });
    desc.textContent = series.length
      ? `${series.length} reading${series.length === 1 ? '' : 's'}. Latest ${describe(latest)}. ` +
        'Shaded bands show ACC/AHA systolic categories.'
      : 'No readings yet.';
    svg.append(title, desc);

    for (const band of BP_CATEGORIES) {
      const top = y(Math.min(yMax, band.sysTo));
      const bottom = y(Math.max(yMin, band.sysFrom));
      if (bottom <= top) continue;
      svg.append(svgEl('rect', { x: pad.left, y: top, width: plotW, height: bottom - top, class: `bp-band band-${band.key}` }));
    }
    for (let v = yMin; v <= yMax; v += 20) {
      svg.append(svgEl('line', { x1: pad.left, x2: pad.left + plotW, y1: y(v), y2: y(v), class: 'chart__grid' }));
      const tick = svgEl('text', { x: pad.left - 6, y: y(v), class: 'chart__tick', 'text-anchor': 'end', 'dominant-baseline': 'central' });
      tick.textContent = String(v);
      svg.append(tick);
    }
    const unit = svgEl('text', { x: 0, y: 2, class: 'chart__tick', 'dominant-baseline': 'hanging' });
    unit.textContent = 'mmHg';
    svg.append(unit);

    if (series.length) {
      const labelIdx = series.length === 1 ? [0] : [0, series.length - 1];
      for (const i of labelIdx) {
        const label = svgEl('text', {
          x: x(i),
          y: height - 6,
          class: 'chart__tick',
          'text-anchor': series.length === 1 ? 'middle' : i === 0 ? 'start' : 'end',
        });
        label.textContent = series[i].time ? formatDateTime(series[i].time) : `Reading ${i + 1}`;
        svg.append(label);
      }
    }

    for (const key of /** @type {const} */ (['systolic', 'diastolic'])) {
      if (series.length > 1) {
        const d = series.map((r, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)} ${y(r[key]).toFixed(1)}`).join('');
        svg.append(svgEl('path', { d, class: `bp-line bp-line--${key === 'systolic' ? 'sys' : 'dia'}` }));
      }
    }
    series.forEach((r, i) => {
      const text = describe(r);
      const g = svgEl('g', { class: 'bp-point', tabindex: 0, role: 'img', 'aria-label': `Reading ${i + 1}: ${text}` });
      g.append(
        svgEl('rect', { x: x(i) - 10, y: y(r.systolic) - 10, width: 20, height: y(r.diastolic) - y(r.systolic) + 20, class: 'chart__hit' }),
        svgEl('circle', { cx: x(i), cy: y(r.systolic), r: 4, class: 'bp-dot bp-dot--sys' }),
        svgEl('circle', { cx: x(i), cy: y(r.diastolic), r: 4, class: 'bp-dot bp-dot--dia' }),
      );
      const show = () => tooltip.show(text, x(i), y(r.systolic));
      g.addEventListener('pointerenter', show);
      g.addEventListener('focus', show);
      g.addEventListener('pointerleave', () => tooltip.hide());
      g.addEventListener('blur', () => tooltip.hide());
      svg.append(g);
    });
    return svg;
  });
}
