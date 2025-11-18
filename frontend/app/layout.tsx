import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { I18nProvider } from './i18n/I18nProvider';
import { cookies } from 'next/headers';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AI Hate Speech Detection',
  description: 'Advanced AI-powered hate speech detection using metaheuristic optimization',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const cookieStore = cookies();
  const saved = cookieStore.get('app_locale')?.value;
  const initialLocale = saved === 'fr' ? 'fr' : 'en';
  return (
    <html lang="en" className="h-full bg-white" suppressHydrationWarning>
      <body className={`${inter.className} min-h-full`}>
        <I18nProvider initialLocale={initialLocale}>
          {children}
        </I18nProvider>
      </body>
    </html>
  );
}
