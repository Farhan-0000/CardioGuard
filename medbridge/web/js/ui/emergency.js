/**
 * Emergency card shown for ESCALATE answers and BP readings in the severe range.
 *
 * Wording is deliberately limited to "get help now" logistics; it adds no new
 * clinical claims beyond the pre-vetted ESCALATE template.
 */

import { h } from './dom.js';
import { icon } from './icons.js';

/** "Do / Don't" logistics for an ESCALATE answer. */
function doAndDont() {
  const list = (items) => h('ul', null, ...items.map((text) => h('li', { text })));
  return h(
    'div',
    { class: 'emergency__lists' },
    h(
      'div',
      null,
      h('h4', { text: 'Do' }),
      list([
        'Call now — do not wait to see if it passes',
        'Tell someone nearby, or ask them to stay with you',
        'Follow the instructions of the emergency operator',
      ]),
    ),
    h(
      'div',
      null,
      h('h4', { text: 'Don’t' }),
      list(['Drive yourself to hospital', 'Take extra medicine unless a clinician tells you to']),
    ),
  );
}

/**
 * @param {{emergencyNumber: string, ambulanceNumber: string, region: string}} config
 * @param {{title?: string, detail?: string | null, lists?: boolean, live?: boolean}} [options]
 *   `detail`: optional pre-vetted server text; `lists`: show the do/don't lists
 *   (off for a severe BP reading, where the server advice is more specific);
 *   `live`: announce the card (role=alert)
 * @returns {HTMLElement}
 */
export function emergencyCard(
  config,
  { title = 'This may be an emergency', detail = null, lists = true, live = false } = {},
) {
  const dial = config.emergencyNumber.replace(/[^\d+]/g, '');
  const ambulance = config.ambulanceNumber.replace(/[^\d+]/g, '');
  const usLine = dial === '911' ? null : h('span', null, ' · In the US call ', h('a', { href: 'tel:911', text: '911' }));
  return h(
    'section',
    { class: 'emergency', role: live ? 'alert' : 'region', 'aria-label': 'Emergency help' },
    h(
      'div',
      { class: 'emergency__head' },
      h('span', { class: 'emergency__icon' }, icon('alert', { size: 20 })),
      h('h3', { text: title }),
    ),
    detail ? h('p', { class: 'emergency__detail', text: detail }) : null,
    h('a', { class: 'emergency__call', href: `tel:${dial}` }, icon('phone', { size: 18 }), `Call ${config.emergencyNumber} now`),
    h(
      'p',
      { class: 'emergency__sub' },
      h('a', { href: `tel:${ambulance}`, text: config.ambulanceNumber }),
      ` for an ambulance (${config.region})`,
      usLine,
    ),
    lists ? doAndDont() : null,
  );
}
