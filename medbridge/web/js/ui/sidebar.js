/**
 * Conversation sidebar: grouped list (Today / Previous 7 days / Older),
 * title search, inline rename and delete.
 */

import { groupConversations } from '../format.js';
import { byId, h } from './dom.js';
import { icon } from './icons.js';

/**
 * @typedef {import('../store.js').Conversation} Conversation
 * @param {{store: ReturnType<typeof import('../store.js').createStore>,
 *          onDelete: (conversation: Conversation) => void,
 *          onRenamed: (conversation: Conversation) => void}} deps
 */
export function createSidebar({ store, onDelete, onRenamed }) {
  const list = byId('conv-list');
  const search = /** @type {HTMLInputElement} */ (byId('conv-search'));
  let activeId = /** @type {string | null} */ (null);

  /**
   * Swap a conversation link for an inline rename field.
   * @param {HTMLElement} row
   * @param {Conversation} conversation
   */
  const startRename = (row, conversation) => {
    const input = /** @type {HTMLInputElement} */ (h('input', {
      class: 'conv__input',
      type: 'text',
      maxlength: 80,
      'aria-label': 'Conversation title',
    }));
    input.value = conversation.title;
    let done = false;
    /** @param {boolean} save */
    const finish = (save) => {
      if (done) return;
      done = true;
      const title = input.value.replace(/\s+/g, ' ').trim();
      if (save && title && title !== conversation.title) {
        const updated = store.updateConversation(conversation.id, { title });
        if (updated) onRenamed(updated);
      }
      render(activeId);
      /** @type {HTMLElement | null} */ (list.querySelector(`[data-id="${CSS.escape(conversation.id)}"] .conv__link`))?.focus();
    };
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        finish(true);
      } else if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        finish(false);
      }
    });
    input.addEventListener('blur', () => finish(true));
    row.replaceChildren(input);
    input.focus();
    input.select();
  };

  /** @param {Conversation} conversation */
  const item = (conversation) => {
    const row = h('li', { class: `conv${conversation.id === activeId ? ' is-active' : ''}`, 'data-id': conversation.id });
    const link = h('a', {
      class: 'conv__link',
      href: `#/c/${encodeURIComponent(conversation.id)}`,
      title: conversation.title,
      'aria-current': conversation.id === activeId ? 'page' : null,
    }, h('span', { class: 'conv__title', text: conversation.title }));
    const rename = h('button', { type: 'button', class: 'icon-btn icon-btn--xs', 'aria-label': `Rename “${conversation.title}”` }, icon('pencil', { size: 15 }));
    rename.addEventListener('click', () => startRename(row, conversation));
    const remove = h('button', { type: 'button', class: 'icon-btn icon-btn--xs', 'aria-label': `Delete “${conversation.title}”` }, icon('trash', { size: 15 }));
    remove.addEventListener('click', () => onDelete(conversation));
    row.append(link, h('div', { class: 'conv__tools' }, rename, remove));
    return row;
  };

  /** @param {string | null} id - active conversation id */
  function render(id) {
    activeId = id;
    const query = search.value.trim().toLowerCase();
    const conversations = store
      .listConversations()
      .filter((c) => !query || c.title.toLowerCase().includes(query));
    const groups = groupConversations(conversations);
    if (!groups.length) {
      list.replaceChildren(h('p', { class: 'conv-empty', text: query ? 'No conversations match your search.' : 'Your conversations will appear here.' }));
      return;
    }
    list.replaceChildren(
      ...groups.map((group) =>
        h('section', { class: 'conv-group', 'aria-label': group.label },
          h('h2', { class: 'conv-group__label', text: group.label }),
          h('ul', { class: 'conv-group__list' }, ...group.items.map(item))),
      ),
    );
  }

  search.addEventListener('input', () => render(activeId));
  search.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && search.value) {
      event.stopPropagation();
      search.value = '';
      render(activeId);
    }
  });

  return { render };
}
