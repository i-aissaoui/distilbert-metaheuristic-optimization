'use client'

import { useState } from 'react'
import { PredictionResponse } from '@/lib/api'
import PredictionForm from '@/app/components/PredictionForm'
import PredictionResult from '@/app/components/PredictionResult'
import ModelComparison from '@/app/components/ModelComparison'
import Training from '@/app/components/Training'
import { Brain, Github, Info } from 'lucide-react'

export default function Home() {
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null)
  const [activeTab, setActiveTab] = useState<'analyze' | 'performance' | 'optimization'>('analyze')

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">AI Hate Speech Detection</h1>
                <p className="text-sm text-gray-600">Multi-Algorithm Optimized DistilBERT</p>
              </div>
            </div>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors"
            >
              <Github className="h-5 w-5" />
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Info Banner */}
        <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-lg flex items-start gap-3">
          <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-blue-900">
            <p className="font-medium mb-1">About This System</p>
            <p>
              This AI-powered system uses a DistilBERT model optimized with advanced metaheuristic algorithms 
              (PSO, Genetic Algorithm, Bayesian Optimization) to detect hate speech with high accuracy. 
              These algorithms automatically find the best hyperparameters across an expanded search space (100× wider learning rate range) 
              to maximize hate speech detection performance and minimize false positives.
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-6 flex gap-2 border-b overflow-x-auto">
          <button
            onClick={() => setActiveTab('analyze')}
            className={`px-6 py-3 font-medium transition-colors whitespace-nowrap ${
              activeTab === 'analyze'
                ? 'border-b-2 border-primary text-primary'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Analyze
          </button>
          <button
            onClick={() => setActiveTab('performance')}
            className={`px-6 py-3 font-medium transition-colors whitespace-nowrap ${
              activeTab === 'performance'
                ? 'border-b-2 border-primary text-primary'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Performance
          </button>
          <button
            onClick={() => setActiveTab('optimization')}
            className={`px-6 py-3 font-medium transition-colors whitespace-nowrap ${
              activeTab === 'optimization'
                ? 'border-b-2 border-primary text-primary'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Optimization
          </button>
        </div>

        {/* Content */}
        {activeTab === 'analyze' ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <PredictionForm onPrediction={setPredictionResult} />
            </div>
            <div>
              {predictionResult ? (
                <PredictionResult result={predictionResult} />
              ) : (
                <div className="h-full flex items-center justify-center p-8 border-2 border-dashed border-gray-300 rounded-lg">
                  <div className="text-center text-gray-500">
                    <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p className="font-medium">No Analysis Yet</p>
                    <p className="text-sm">Enter a message to see if it's spam</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : activeTab === 'performance' ? (
          <ModelComparison />
        ) : (
          <Training mode="optimization" onTrainingComplete={() => {}} />
        )}

        {/* Footer Info */}
        <div className="mt-12 p-6 bg-white rounded-lg border">
          <h3 className="text-lg font-semibold mb-3">Classification Labels</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { label: 'Spam', icon: '🚫', desc: 'Unwanted or malicious message', color: 'red' },
              { label: 'Ham (Not Spam)', icon: '✅', desc: 'Legitimate message', color: 'green' },
            ].map((item) => (
              <div key={item.label} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                <span className="text-2xl">{item.icon}</span>
                <div>
                  <p className="font-medium text-sm">{item.label}</p>
                  <p className="text-xs text-gray-600">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 border-t bg-white">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-sm text-gray-600">
            Built with Next.js, FastAPI, and Multi-Algorithm Optimized DistilBERT (PSO • GA • Bayesian)
          </p>
        </div>
      </footer>
    </div>
  )
}
