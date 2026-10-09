/**
 * One write at a time per conversation.
 *
 * A chat turn computes the patient's context snapshot from the state it loaded
 * when it started and saves it when it ends. A second write to the same
 * conversation in the meantime (a manual BP reading, another message, a
 * regenerate) could be overwritten by that snapshot, so the UI never starts
 * one while another is in flight. That includes a turn the user stopped or
 * whose connection dropped: the server deliberately finishes and saves it, so
 * the conversation stays locked until the saved answer shows up in the history
 * (or a time limit passes).
 */

/** @typedef {'turn' | 'regenerate' | 'bp' | 'finishing'} WriteKind */

/**
 * Message shown when a write is blocked by `kind`.
 * @param {WriteKind | string | null} kind
 */
export function busyMessage(kind) {
  switch (kind) {
    case 'finishing':
      return 'MedBridge is still finishing the previous answer for this conversation. Please wait a moment.';
    case 'bp':
      return 'Your blood pressure reading is still being saved. Please wait a moment.';
    case 'regenerate':
    case 'turn':
    default:
      return 'Please wait until the current answer has finished.';
  }
}

/** Per-conversation write lock with change notifications. */
export function createWriteGuard() {
  /** @type {Map<string, {kind: WriteKind}>} */
  const held = new Map();
  /** @type {Set<(id: string) => void>} */
  const listeners = new Set();
  /** @param {string} id */
  const emit = (id) => {
    for (const listener of listeners) listener(id);
  };

  return {
    /**
     * Kind of the write in progress for `id`, or null when it is free.
     * @param {string | null | undefined} id
     * @returns {WriteKind | null}
     */
    activeKind(id) {
      return (id && held.get(id)?.kind) || null;
    },
    /** @param {string | null | undefined} id */
    isBusy(id) {
      return Boolean(id) && held.has(/** @type {string} */ (id));
    },
    /**
     * Take the lock for `id`. Returns a handle, or null when another write holds it.
     * @param {string} id
     * @param {WriteKind} kind
     * @returns {{release: () => void, relabel: (kind: WriteKind) => void} | null}
     */
    acquire(id, kind) {
      if (!id || held.has(id)) return null;
      const entry = { kind };
      held.set(id, entry);
      emit(id);
      return {
        release() {
          if (held.get(id) !== entry) return;
          held.delete(id);
          emit(id);
        },
        relabel(next) {
          if (held.get(id) !== entry || entry.kind === next) return;
          entry.kind = next;
          emit(id);
        },
      };
    },
    /**
     * @param {(id: string) => void} listener
     * @returns {() => void} unsubscribe
     */
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

/**
 * Number of assistant messages in a history payload / message list.
 * @param {unknown} messages
 */
export function countAssistantMessages(messages) {
  return Array.isArray(messages) ? messages.filter((m) => m && m.role === 'assistant').length : 0;
}

/**
 * Poll the history until the server has saved a turn the client stopped
 * (one more assistant message than `baseline`).
 * @param {{
 *   fetchHistory: () => Promise<any>,
 *   baseline: number,
 *   intervalMs?: number,
 *   maxIntervalMs?: number,
 *   timeoutMs?: number,
 *   sleep?: (ms: number) => Promise<void>,
 *   now?: () => number,
 *   isCancelled?: () => boolean,
 * }} options
 * @returns {Promise<'saved' | 'gone' | 'timeout' | 'cancelled'>}
 */
export async function waitForSavedTurn({
  fetchHistory,
  baseline,
  intervalMs = 1500,
  maxIntervalMs = 5000,
  timeoutMs = 90_000,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  now = () => Date.now(),
  isCancelled = () => false,
}) {
  const deadline = now() + timeoutMs;
  let delay = intervalMs;
  for (;;) {
    if (isCancelled()) return 'cancelled';
    try {
      const data = await fetchHistory();
      if (countAssistantMessages(data?.messages) > baseline) return 'saved';
    } catch (error) {
      // The conversation was erased or access is gone: nothing left to protect.
      if (/** @type {any} */ (error)?.sessionUnavailable) return 'gone';
      // Network error, 5xx or 429: keep waiting until the deadline.
    }
    if (isCancelled()) return 'cancelled';
    if (now() + delay > deadline) return 'timeout';
    await sleep(delay);
    delay = Math.min(maxIntervalMs, Math.round(delay * 1.5));
  }
}
