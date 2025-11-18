export type Locale = 'en' | 'fr'

export const messages = {
  en: {
    app: {
      title: 'AI Hate Speech Detection',
      subtitle: 'Powered by Metaheuristic-Optimized DistilBERT',
    },
    nav: {
      home: 'Home',
      analyze: 'Analyze',
      performance: 'Performance',
      optimization: 'Optimization',
      lang: 'Language',
    },
    home: {
      hero: {
        title: 'Advanced AI-Powered Hate Speech Detection',
        subtitle: 'Leveraging state-of-the-art NLP and metaheuristic optimization to identify and analyze hate speech with unprecedented accuracy.',
        try: 'Try It Now',
        learn: 'Learn More',
      },
      features: {
        title: 'Key Features',
        analyze: {
          title: 'Text Analysis',
          desc: 'Analyze text in real-time to detect hate speech and offensive content with high accuracy.',
        },
        performance: {
          title: 'Performance Metrics',
          desc: 'Compare model performance with detailed metrics and visualizations.',
        },
        optimization: {
          title: 'Optimization',
          desc: 'Fine-tune model parameters using advanced optimization algorithms.',
        },
      },
      how: {
        title: 'How It Works',
        step1: { title: 'Input Your Text', desc: 'Enter any text content into our analysis tool to detect potential hate speech or offensive language.' },
        step2: { title: 'Advanced Analysis', desc: 'Our AI model processes the text using optimized DistilBERT architecture for accurate classification.' },
        step3: { title: 'Get Insights', desc: 'Receive detailed results including confidence scores and explanations for the classification.' },
      },
      footer: 'All rights reserved.',
    },
  },
  fr: {
    app: {
      title: "Détection d'Harcèlement par IA",
      subtitle: 'Propulsé par DistilBERT optimisé par métaheuristiques',
    },
    nav: {
      home: 'Accueil',
      analyze: 'Analyse',
      performance: 'Performance',
      optimization: 'Optimisation',
      lang: 'Langue',
    },
    home: {
      hero: {
        title: 'Détection avancée de discours haineux par IA',
        subtitle: "Exploitez le NLP de pointe et l'optimisation métaheuristique pour identifier et analyser les discours haineux avec une précision inégalée.",
        try: 'Essayer maintenant',
        learn: 'En savoir plus',
      },
      features: {
        title: 'Fonctionnalités clés',
        analyze: {
          title: 'Analyse de texte',
          desc: 'Analysez le texte en temps réel pour détecter les discours haineux et le contenu offensant avec une grande précision.',
        },
        performance: {
          title: 'Mesures de performance',
          desc: 'Comparez les performances du modèle avec des métriques détaillées et des visualisations.',
        },
        optimization: {
          title: 'Optimisation',
          desc: 'Ajustez les paramètres du modèle avec des algorithmes avancés.',
        },
      },
      how: {
        title: 'Comment ça marche',
        step1: { title: 'Saisissez votre texte', desc: "Entrez n'importe quel texte dans notre outil pour détecter un éventuel discours haineux." },
        step2: { title: 'Analyse avancée', desc: 'Notre modèle IA traite le texte avec DistilBERT optimisé pour une classification précise.' },
        step3: { title: 'Obtenez des insights', desc: 'Recevez des résultats détaillés, y compris les scores de confiance et des explications.' },
      },
      footer: 'Tous droits réservés.',
    },
  },
} as const

export function isLocale(x: string): x is Locale {
  return x === 'en' || x === 'fr'
}

export function translate(locale: Locale, key: string): string {
  const parts = key.split('.')
  let ref: any = messages[locale]
  for (const p of parts) {
    if (ref && typeof ref === 'object' && p in ref) {
      ref = ref[p]
    } else {
      return key
    }
  }
  return typeof ref === 'string' ? ref : key
}
