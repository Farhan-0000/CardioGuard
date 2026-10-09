/**
 * Client-side persistence with an in-memory fallback.
 *
 * Only presentation state lives in the browser: the conversation list
 * ({id, token, title, createdAt, updatedAt}), theme, patient/clinician mode,
 * consent flag and sidebar state. The server never stores conversation
 * titles. Every storage access is wrapped in try/catch because localStorage
 * can be missing or throw (private mode, blocked site data, sandboxed
 * frames); values are then kept in memory for the lifetime of the page.
 */

const PREFIX = 'medbridge.';
export const CONSENT_VERSION = '1';

/**
 * Resolve a Storage object without throwing.
 * @param {'localStorage' | 'sessionStorage'} name
 * @returns {Storage | null}
 */
export function resolveStorage(name) {
  try {
    const storage = globalThis[name];
    return storage && typeof storage.getItem === 'function' ? storage : null;
  } catch {
    return null;
  }
}

/**
 * Key/value store that never throws.
 * @param {Storage | null} storage
 */
export function createSafeStorage(storage) {
  /** @type {Map<string, string>} */
  const memory = new Map();
  return {
    /** @param {string} key */
    get(key) {
      try {
        const value = storage ? storage.getItem(PREFIX + key) : null;
        if (value !== null && value !== undefined) return value;
      } catch {
        // fall through to memory
      }
      return memory.has(key) ? /** @type {string} */ (memory.get(key)) : null;
    },
    /** @param {string} key @param {string} value */
    set(key, value) {
      memory.set(key, value);
      try {
        if (storage) storage.setItem(PREFIX + key, value);
      } catch {
        // quota exceeded / blocked: memory copy is kept
      }
    },
    /** @param {string} key */
    remove(key) {
      memory.delete(key);
      try {
        if (storage) storage.removeItem(PREFIX + key);
      } catch {
        // ignore
      }
    },
  };
}

/**
 * @typedef {{id: string, token: string | null, title: string, createdAt: number, updatedAt: number}} Conversation
 */

/** @param {any} item @returns {item is Conversation} */
function isConversation(item) {
  return Boolean(item) && typeof item.id === 'string' && item.id.length > 0;
}

/**
 * Application store (conversations + preferences).
 * @param {Storage | null} [storage]
 */
export function createStore(storage = resolveStorage('localStorage')) {
  const kv = createSafeStorage(storage);

  /** @returns {Conversation[]} */
  function readConversations() {
    const raw = kv.get('conversations');
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(isConversation).map((c) => ({
        id: c.id,
        token: typeof c.token === 'string' ? c.token : null,
        title: typeof c.title === 'string' && c.title ? c.title : 'New conversation',
        createdAt: Number(c.createdAt) || 0,
        updatedAt: Number(c.updatedAt) || Number(c.createdAt) || 0,
      }));
    } catch {
      return [];
    }
  }

  /** @param {Conversation[]} list */
  function writeConversations(list) {
    kv.set('conversations', JSON.stringify(list));
  }

  return {
    /** @returns {Conversation[]} newest first */
    listConversations() {
      return readConversations().sort((a, b) => b.updatedAt - a.updatedAt);
    },
    /** @param {string} id */
    getConversation(id) {
      return readConversations().find((c) => c.id === id) ?? null;
    },
    /** @param {Conversation} conversation */
    saveConversation(conversation) {
      const list = readConversations().filter((c) => c.id !== conversation.id);
      list.push(conversation);
      writeConversations(list);
    },
    /** @param {string} id @param {Partial<Conversation>} patch */
    updateConversation(id, patch) {
      const list = readConversations();
      const item = list.find((c) => c.id === id);
      if (!item) return null;
      Object.assign(item, patch);
      writeConversations(list);
      return item;
    },
    /** @param {string} id */
    removeConversation(id) {
      writeConversations(readConversations().filter((c) => c.id !== id));
    },
    /** @returns {'light' | 'dark' | null} */
    getTheme() {
      const value = kv.get('theme');
      return value === 'light' || value === 'dark' ? value : null;
    },
    /** @param {'light' | 'dark'} theme */
    setTheme(theme) {
      kv.set('theme', theme);
    },
    /** @returns {'patient' | 'clinician'} */
    getMode() {
      return kv.get('mode') === 'clinician' ? 'clinician' : 'patient';
    },
    /** @param {'patient' | 'clinician'} mode */
    setMode(mode) {
      kv.set('mode', mode);
    },
    hasConsent() {
      return kv.get('consent') === CONSENT_VERSION;
    },
    setConsent() {
      kv.set('consent', CONSENT_VERSION);
    },
    getSidebarCollapsed() {
      return kv.get('sidebarCollapsed') === '1';
    },
    /** @param {boolean} collapsed */
    setSidebarCollapsed(collapsed) {
      kv.set('sidebarCollapsed', collapsed ? '1' : '0');
    },
  };
}
