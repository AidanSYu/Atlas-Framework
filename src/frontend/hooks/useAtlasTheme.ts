'use client';

import { useCallback, useEffect, useState } from 'react';

export type AtlasTheme = 'dark' | 'light';

const THEME_STORAGE_KEY = 'atlas-theme';

function normalizeTheme(value: string | null): AtlasTheme {
  return value === 'light' ? 'light' : 'dark';
}

export function applyAtlasTheme(theme: AtlasTheme): void {
  if (typeof document === 'undefined') return;
  document.documentElement.classList.toggle('light', theme === 'light');
  document.documentElement.dataset.theme = theme;
}

export function getStoredAtlasTheme(): AtlasTheme {
  if (typeof window === 'undefined') return 'dark';
  try {
    return normalizeTheme(window.localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return 'dark';
  }
}

export function useAtlasTheme() {
  const [theme, setThemeState] = useState<AtlasTheme>('dark');

  useEffect(() => {
    const savedTheme = getStoredAtlasTheme();
    setThemeState(savedTheme);
    applyAtlasTheme(savedTheme);

    const handleStorage = (event: StorageEvent) => {
      if (event.key && event.key !== THEME_STORAGE_KEY) return;
      const nextTheme = normalizeTheme(event.newValue);
      setThemeState(nextTheme);
      applyAtlasTheme(nextTheme);
    };

    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const setTheme = useCallback((nextTheme: AtlasTheme | ((currentTheme: AtlasTheme) => AtlasTheme)) => {
    setThemeState((currentTheme) => {
      const resolvedTheme = typeof nextTheme === 'function' ? nextTheme(currentTheme) : nextTheme;
      applyAtlasTheme(resolvedTheme);
      try {
        window.localStorage.setItem(THEME_STORAGE_KEY, resolvedTheme);
      } catch {
        // Ignore storage failures and still apply the in-memory theme.
      }
      return resolvedTheme;
    });
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'));
  }, [setTheme]);

  return {
    theme,
    setTheme,
    toggleTheme,
  };
}
