import Script from 'next/script';

const initThemeScript = `
  try {
    var storedTheme = localStorage.getItem('atlas-theme');
    var theme = storedTheme === 'light' ? 'light' : 'dark';
    document.documentElement.classList.toggle('light', theme === 'light');
    document.documentElement.dataset.theme = theme;
  } catch (error) {
    document.documentElement.classList.remove('light');
    document.documentElement.dataset.theme = 'dark';
  }
`;

export function ThemeInitializerScript() {
  return (
    <Script id="atlas-theme-init" strategy="beforeInteractive">
      {initThemeScript}
    </Script>
  );
}
