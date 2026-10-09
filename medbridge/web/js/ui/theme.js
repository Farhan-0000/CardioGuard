/**
 * Light/dark theme: follows prefers-color-scheme until the user picks one;
 * the explicit choice is stored and applied as <html data-theme="...">.
 */

import { icon } from './icons.js';

const DARK = window.matchMedia('(prefers-color-scheme: dark)');

/**
 * Wire a toggle button.
 * @param {HTMLElement} button
 * @param {{getTheme: () => 'light' | 'dark' | null, setTheme: (t: 'light' | 'dark') => void}} store
 */
export function initThemeToggle(button, store) {
  const effective = () => store.getTheme() ?? (DARK.matches ? 'dark' : 'light');
  const apply = () => {
    const saved = store.getTheme();
    if (saved) document.documentElement.dataset.theme = saved;
    else delete document.documentElement.dataset.theme;
    const next = effective() === 'dark' ? 'light' : 'dark';
    button.setAttribute('aria-label', `Switch to ${next} theme`);
    button.dataset.tooltip = next === 'dark' ? 'Dark theme' : 'Light theme';
    button.replaceChildren(icon(next === 'dark' ? 'moon' : 'sun'));
  };
  button.addEventListener('click', () => {
    store.setTheme(effective() === 'dark' ? 'light' : 'dark');
    apply();
  });
  DARK.addEventListener('change', apply);
  apply();
}
