"use client";
import Navbar from '@/app/components/Navbar';
import { Brain, BarChart, Zap, Search } from 'lucide-react';
import Link from 'next/link';
import { useI18n } from '@/app/i18n/I18nProvider';

export default function HomePage() {
  const { t } = useI18n();
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Navbar />
      
      <main className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <section className="text-center max-w-4xl mx-auto mb-20">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">{t('home.hero.title')}</h1>
          <p className="text-xl text-gray-600 mb-8">{t('home.hero.subtitle')}</p>
          <div className="flex gap-4 justify-center">
            <Link 
              href="/analyze"
              className="px-6 py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary/90 transition-colors"
            >
              {t('home.hero.try')}
            </Link>
            <Link 
              href="#features"
              className="px-6 py-3 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition-colors"
            >
              {t('home.hero.learn')}
            </Link>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="max-w-6xl mx-auto mb-20">
          <h2 className="text-3xl font-bold text-center mb-12">{t('home.features.title')}</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <Search className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">{t('home.features.analyze.title')}</h3>
              <p className="text-gray-600">{t('home.features.analyze.desc')}</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                <BarChart className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">{t('home.features.performance.title')}</h3>
              <p className="text-gray-600">{t('home.features.performance.desc')}</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">{t('home.features.optimization.title')}</h3>
              <p className="text-gray-600">{t('home.features.optimization.desc')}</p>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="max-w-4xl mx-auto mb-20">
          <h2 className="text-3xl font-bold text-center mb-12">{t('home.how.title')}</h2>
          <div className="space-y-8">
            <div className="flex gap-6">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary text-white flex items-center justify-center font-bold">1</div>
              <div>
                <h3 className="text-xl font-semibold mb-2">{t('home.how.step1.title')}</h3>
                <p className="text-gray-600">{t('home.how.step1.desc')}</p>
              </div>
            </div>
            <div className="flex gap-6">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary text-white flex items-center justify-center font-bold">2</div>
              <div>
                <h3 className="text-xl font-semibold mb-2">{t('home.how.step2.title')}</h3>
                <p className="text-gray-600">{t('home.how.step2.desc')}</p>
              </div>
            </div>
            <div className="flex gap-6">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary text-white flex items-center justify-center font-bold">3</div>
              <div>
                <h3 className="text-xl font-semibold mb-2">{t('home.how.step3.title')}</h3>
                <p className="text-gray-600">{t('home.how.step3.desc')}</p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="bg-white border-t py-8">
        <div className="container mx-auto px-4 text-center text-gray-600">
          <p>© {new Date().getFullYear()} {t('app.title')}. {t('home.footer')}</p>
        </div>
      </footer>
    </div>
  );
}
