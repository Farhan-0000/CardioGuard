/**
 * Right-side source drawer opened from citation chips and the Sources list.
 */

import { safeUrl } from '../markdown.js';
import { byId, h } from './dom.js';
import { icon } from './icons.js';

export function createSourceDrawer() {
  const drawer = byId('source-drawer');
  const content = byId('drawer-content');
  const closeButton = byId('drawer-close');
  /** @type {HTMLElement | null} */
  let returnFocus = null;

  const close = () => {
    if (drawer.hidden) return;
    drawer.classList.remove('is-open');
    drawer.hidden = true;
    document.body.classList.remove('drawer-open');
    if (returnFocus && document.contains(returnFocus)) returnFocus.focus();
    returnFocus = null;
  };

  closeButton.addEventListener('click', close);

  return {
    close,
    get isOpen() {
      return !drawer.hidden;
    },
    /**
     * @param {any} citation - CitationResponse
     * @param {HTMLElement} trigger
     */
    open(citation, trigger) {
      returnFocus = trigger;
      const n = String(citation.marker ?? '').replace(/[^\d]/g, '');
      const facts = [
        ['Organisation', citation.org],
        ['Year', citation.year],
        ['Section', citation.section],
        ['Page', citation.page],
        ['Document id', citation.guideline_id],
      ].filter(([, v]) => v !== null && v !== undefined && v !== '');
      const href = citation.url ? safeUrl(citation.url) : null;
      byId('drawer-title').textContent = `Source ${n}`;
      const nodes = [
        h('p', { class: 'drawer__source', text: citation.source || '' }),
        h('h3', { class: 'drawer__heading', text: citation.title || citation.source || 'Clinical guideline' }),
        facts.length
          ? h('dl', { class: 'kv kv--drawer' }, ...facts.flatMap(([k, v]) => [h('dt', { text: String(k) }), h('dd', { text: String(v) })]))
          : null,
        citation.excerpt
          ? h('figure', { class: 'excerpt' }, h('figcaption', { text: 'Excerpt used as evidence' }), h('blockquote', { text: citation.excerpt }))
          : null,
        href
          ? h('a', { class: 'btn btn--ghost drawer__link', href, target: '_blank', rel: 'noopener noreferrer' }, 'Open the guideline', icon('chevronDown', { size: 14, className: 'icon--rot' }))
          : null,
        h('p', { class: 'drawer__note', text: 'Excerpts come from the indexed guideline text and may be shortened.' }),
      ];
      content.replaceChildren(...nodes.filter((node) => node !== null));
      drawer.hidden = false;
      document.body.classList.add('drawer-open');
      requestAnimationFrame(() => drawer.classList.add('is-open'));
      closeButton.focus();
    },
  };
}
