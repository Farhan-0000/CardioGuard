// Unit tests for the localStorage wrapper with in-memory fallback.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createSafeStorage, createStore, resolveStorage } from '../js/store.js';

class MemoryStorage {
  constructor() {
    this.map = new Map();
  }
  getItem(k) {
    return this.map.has(k) ? this.map.get(k) : null;
  }
  setItem(k, v) {
    this.map.set(k, String(v));
  }
  removeItem(k) {
    this.map.delete(k);
  }
}

class ThrowingStorage {
  getItem() {
    throw new Error('SecurityError');
  }
  setItem() {
    throw new Error('QuotaExceededError');
  }
  removeItem() {
    throw new Error('SecurityError');
  }
}

test('conversations are saved, listed newest first, updated and removed', () => {
  const store = createStore(new MemoryStorage());
  store.saveConversation({ id: 'a', token: 't1', title: 'First', createdAt: 1, updatedAt: 10 });
  store.saveConversation({ id: 'b', token: 't2', title: 'Second', createdAt: 2, updatedAt: 20 });
  assert.deepEqual(store.listConversations().map((c) => c.id), ['b', 'a']);
  store.updateConversation('a', { title: 'Renamed', updatedAt: 30 });
  assert.equal(store.listConversations()[0].title, 'Renamed');
  assert.equal(store.getConversation('b').token, 't2');
  store.removeConversation('b');
  assert.equal(store.getConversation('b'), null);
  assert.equal(store.updateConversation('missing', { title: 'x' }), null);
});

test('a throwing storage falls back to memory', () => {
  const store = createStore(new ThrowingStorage());
  store.saveConversation({ id: 'x', token: null, title: 'T', createdAt: 1, updatedAt: 1 });
  assert.equal(store.listConversations().length, 1);
  store.setTheme('dark');
  assert.equal(store.getTheme(), 'dark');
  assert.equal(store.hasConsent(), false);
  store.setConsent();
  assert.equal(store.hasConsent(), true);
});

test('no storage at all (null) still works', () => {
  const store = createStore(null);
  store.setMode('clinician');
  assert.equal(store.getMode(), 'clinician');
  store.setSidebarCollapsed(true);
  assert.equal(store.getSidebarCollapsed(), true);
});

test('corrupt or foreign data is ignored safely', () => {
  const storage = new MemoryStorage();
  storage.setItem('medbridge.conversations', '{not json');
  assert.deepEqual(createStore(storage).listConversations(), []);
  storage.setItem('medbridge.conversations', JSON.stringify([{ id: 'ok' }, { nope: 1 }, null, 'x']));
  const list = createStore(storage).listConversations();
  assert.equal(list.length, 1);
  assert.equal(list[0].title, 'New conversation');
  assert.equal(list[0].token, null);
});

test('theme and mode only accept known values', () => {
  const storage = new MemoryStorage();
  storage.setItem('medbridge.theme', 'neon');
  storage.setItem('medbridge.mode', 'admin');
  const store = createStore(storage);
  assert.equal(store.getTheme(), null);
  assert.equal(store.getMode(), 'patient');
});

test('createSafeStorage remove() never throws', () => {
  const kv = createSafeStorage(new ThrowingStorage());
  kv.set('k', 'v');
  assert.equal(kv.get('k'), 'v');
  kv.remove('k');
  assert.equal(kv.get('k'), null);
});

test('resolveStorage returns null when the global accessor throws', () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'sessionStorage');
  Object.defineProperty(globalThis, 'sessionStorage', {
    configurable: true,
    get() {
      throw new Error('SecurityError');
    },
  });
  try {
    assert.equal(resolveStorage('sessionStorage'), null);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'sessionStorage', descriptor);
    else delete globalThis.sessionStorage;
  }
});
