'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/app/components/ui/card'
import { Button } from '@/app/components/ui/button'
import { api, PredictionResponse } from '@/lib/api'
import { Loader2, Send, Shuffle } from 'lucide-react'

interface PredictionFormProps {
  onPrediction: (result: PredictionResponse) => void
}

export default function PredictionForm({ onPrediction }: PredictionFormProps) {
  const [text, setText] = useState('')
  const [selectedModel, setSelectedModel] = useState<'baseline' | 'pso' | 'ga' | 'bayesian'>('pso')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [trueLabel, setTrueLabel] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!text.trim()) {
      setError('Please enter a statement to analyze')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const result = await api.predict({
        text: text.trim(),
        use_optimized: selectedModel !== 'baseline',
        model_type: selectedModel === 'baseline' ? 'pso' : selectedModel as 'pso' | 'ga' | 'bayesian',
      })
      onPrediction(result)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze statement. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const hateSpeechExamples = [
    { text: "I hate all people from that country, they should go back where they came from!", label: "hate" },
    { text: "Have a wonderful day! Hope everything goes well for you.", label: "not-hate" },
    { text: "You're so stupid and ugly, nobody will ever love you", label: "hate" },
    { text: "Thanks for your help with the project. I really appreciate it!", label: "not-hate" },
    { text: "All [group] are criminals and should be eliminated", label: "hate" },
    { text: "Let's work together to make our community better for everyone.", label: "not-hate" },
  ]

  const loadRandomExample = () => {
    const randomExample = hateSpeechExamples[Math.floor(Math.random() * hateSpeechExamples.length)]
    setText(randomExample.text)
    setTrueLabel(randomExample.label)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Analyze Message</CardTitle>
        <CardDescription>
          Enter a message to detect if it contains hate speech using our AI model
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value)
                setTrueLabel(null)
              }}
              placeholder="Enter a message to analyze..."
              className="w-full min-h-[120px] p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary resize-none"
              disabled={loading}
            />
            {trueLabel && (
              <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm">
                <span className="font-medium">True Label:</span> <span className={trueLabel === 'hate' ? 'text-red-600 font-semibold' : 'text-green-600 font-semibold'}>{trueLabel === 'hate' ? '🚫 Hate Speech' : '✅ Not Hate'}</span>
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Select Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as any)}
              className="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              disabled={loading}
            >
              <option value="baseline">Baseline Model</option>
              <option value="pso">PSO Optimized Model</option>
              <option value="ga">GA Optimized Model</option>
              <option value="bayesian">Bayesian Optimized Model</option>
            </select>
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-md text-red-600 text-sm">
              {error}
            </div>
          )}

          <Button type="submit" disabled={loading} className="w-full">
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="mr-2 h-4 w-4" />
                Analyze Message
              </>
            )}
          </Button>
        </form>

        <div className="mt-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-medium text-muted-foreground">Try an example:</p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={loadRandomExample}
              disabled={loading}
            >
              <Shuffle className="mr-2 h-3 w-3" />
              Random Example
            </Button>
          </div>
          <div className="space-y-2">
            {hateSpeechExamples.slice(0, 3).map((example, index) => (
              <button
                key={index}
                onClick={() => {
                  setText(example.text)
                  setTrueLabel(example.label)
                }}
                className="w-full text-left p-2 text-sm bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
                disabled={loading}
              >
                <span className={example.label === 'hate' ? 'text-red-600 font-medium' : 'text-green-600 font-medium'}>
                  {example.label === 'hate' ? '🚫' : '✅'}
                </span> {example.text.substring(0, 80)}...
              </button>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
