/**
 * Hand-drawn 24x24 stroke icons (original geometry, no icon library).
 * Icons inherit `currentColor` and are hidden from assistive technology;
 * the owning button always carries an aria-label.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

/** @type {Record<string, string[]>} path data per icon */
const PATHS = {
  sidebar: ['M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z', 'M9.5 4v16'],
  compose: [
    'M11 4.5H6.5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V13',
    'M17.6 3.9a1.9 1.9 0 0 1 2.7 2.7L12.5 14.4 9 15.2l.8-3.5z',
  ],
  search: ['M10.8 18a7.2 7.2 0 1 0 0-14.4 7.2 7.2 0 0 0 0 14.4z', 'M16 16l4.5 4.5'],
  pencil: ['M4.5 19.5h4L19 9l-4-4L4.5 15.5z', 'M13 7l4 4'],
  trash: ['M4.5 7h15', 'M9.5 7V4.5h5V7', 'M6.5 7l.9 12.5h9.2L17.5 7', 'M10.3 11v5M13.7 11v5'],
  sun: [
    'M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
    'M12 2.5v2M12 19.5v2M4.6 4.6 6 6M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4',
  ],
  moon: ['M19.5 14.6A7.8 7.8 0 0 1 9.4 4.5a7.8 7.8 0 1 0 10.1 10.1z'],
  pulse: ['M3 12h4l2.5-6 5 12 2.5-6h4'],
  copy: ['M10 9h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1z', 'M5.5 15H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v.5'],
  check: ['M5 12.5l4.5 4.5L19 7.5'],
  refresh: ['M19.5 12a7.5 7.5 0 1 1-2.2-5.3', 'M19.5 4.5V9H15'],
  thumbUp: [
    'M7.5 10.5v9h-3v-9z',
    'M7.5 10.5 11 3.8a1.9 1.9 0 0 1 3.4 1.5l-.9 4.2h5a1.8 1.8 0 0 1 1.8 2.2l-1.3 6.4a2.3 2.3 0 0 1-2.3 1.9H7.5',
  ],
  thumbDown: [
    'M7.5 13.5v-9h-3v9z',
    'M7.5 13.5 11 20.2a1.9 1.9 0 0 0 3.4-1.5l-.9-4.2h5a1.8 1.8 0 0 0 1.8-2.2l-1.3-6.4a2.3 2.3 0 0 0-2.3-1.9H7.5',
  ],
  send: ['M12 19V5.5', 'M6 11.5 12 5.5l6 6'],
  close: ['M6 6l12 12M18 6 6 18'],
  shield: ['M12 3 19 6v5.2c0 4.4-2.9 8.1-7 9.8-4.1-1.7-7-5.4-7-9.8V6z', 'M9 12l2.1 2.1L15.2 10'],
  alert: ['M12 4.2 21 19.5H3z', 'M12 10v4.2', 'M12 17v.01'],
  phone: [
    'M6.2 4h3l1.6 4.3-2 1.3a10.6 10.6 0 0 0 5.6 5.6l1.3-2 4.3 1.6v3a2 2 0 0 1-2.1 2A15.4 15.4 0 0 1 4.2 6.1 2 2 0 0 1 6.2 4z',
  ],
  info: ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z', 'M12 11v5.5', 'M12 7.8v.01'],
  chevronDown: ['M6.5 9.5 12 15l5.5-5.5'],
  download: ['M12 4v11', 'M7.5 10.5 12 15l4.5-4.5', 'M5 19.5h14'],
  printer: ['M7 9V4h10v5', 'M6 17H5a1.5 1.5 0 0 1-1.5-1.5v-5A1.5 1.5 0 0 1 5 9h14a1.5 1.5 0 0 1 1.5 1.5v5A1.5 1.5 0 0 1 19 17h-1', 'M7 13.5h10V20H7z'],
  file: ['M14 3.5H7a1.5 1.5 0 0 0-1.5 1.5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8z', 'M14 3.5V8h4.5', 'M9 13h6M9 16.5h6'],
  book: ['M5 5.5A2 2 0 0 1 7 3.5h12v14H7a2 2 0 0 0-2 2z', 'M5 19.5a2 2 0 0 0 2 2h12v-4'],
  help: ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z', 'M9.6 9.3a2.5 2.5 0 1 1 3.4 2.4c-.6.3-1 .8-1 1.5v.6', 'M12 17v.01'],
  lock: ['M6.5 11h11a1.5 1.5 0 0 1 1.5 1.5v6.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 19v-6.5A1.5 1.5 0 0 1 6.5 11z', 'M8.5 11V8a3.5 3.5 0 0 1 7 0v3'],
  logout: ['M14.5 4.5h3a1.5 1.5 0 0 1 1.5 1.5v12a1.5 1.5 0 0 1-1.5 1.5h-3', 'M10 16.5 5.5 12 10 7.5', 'M5.5 12H15'],
  eyeOff: [
    'M3.5 3.5l17 17',
    'M10.3 6.2A9 9 0 0 1 12 6c5 0 8.5 6 8.5 6a15.6 15.6 0 0 1-2.6 3.3',
    'M6.6 7.6C4.6 9.1 3.5 12 3.5 12s3.5 6 8.5 6a8.7 8.7 0 0 0 4-.9',
    'M10 10a2.8 2.8 0 0 0 4 4',
  ],
  user: ['M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8z', 'M4.5 20.5a7.5 7.5 0 0 1 15 0'],
  stop: [],
  plus: ['M12 5v14M5 12h14'],
  chart: ['M4 20h16', 'M7 16v-5M12 16V7M17 16v-8'],
};

/**
 * Build an inline SVG icon.
 * @param {keyof typeof PATHS} name
 * @param {{size?: number, className?: string}} [options]
 * @returns {SVGSVGElement}
 */
export function icon(name, { size = 18, className = '' } = {}) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', String(size));
  svg.setAttribute('height', String(size));
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.setAttribute('class', `icon${className ? ` ${className}` : ''}`);
  if (name === 'stop') {
    const rect = document.createElementNS(SVG_NS, 'rect');
    for (const [k, v] of Object.entries({ x: '7', y: '7', width: '10', height: '10', rx: '2' })) {
      rect.setAttribute(k, v);
    }
    rect.setAttribute('fill', 'currentColor');
    svg.append(rect);
    return svg;
  }
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.8');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  for (const d of PATHS[name] ?? []) {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    svg.append(path);
  }
  return svg;
}

/**
 * Replace the icon inside every `[data-icon]` placeholder in `root`.
 * @param {ParentNode} root
 */
export function hydrateIcons(root) {
  for (const el of root.querySelectorAll('[data-icon]')) {
    const name = /** @type {keyof typeof PATHS} */ (el.getAttribute('data-icon'));
    const size = Number(el.getAttribute('data-icon-size')) || 18;
    el.replaceChildren(icon(name, { size }));
  }
}
