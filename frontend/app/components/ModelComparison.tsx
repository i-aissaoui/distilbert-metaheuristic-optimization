'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/app/components/ui/card'
import { api, ModelInfo } from '@/lib/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { TrendingUp, Award, Zap } from 'lucide-react'

export default function ModelComparison() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const info = await api.getModelInfo()
        setModelInfo(info)
      } catch (error) {
        console.error('Failed to fetch model info:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchModelInfo()
  }, [])

  if (!mounted || loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Loading model information...</p>
        </CardContent>
      </Card>
    )
  }

  if (!modelInfo?.baseline_metrics || !modelInfo?.optimized_metrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Model performance data not available. Please train the models first.
          </p>
        </CardContent>
      </Card>
    )
  }

  const comparisonData = [
    {
      metric: 'Accuracy',
      Baseline: (modelInfo.baseline_metrics.accuracy * 100).toFixed(2),
      PSO: modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.accuracy * 100).toFixed(2) : 'N/A',
      GA: 'N/A',
      Bayesian: 'N/A',
    },
    {
      metric: 'F1 Score',
      Baseline: (modelInfo.baseline_metrics.f1_weighted * 100).toFixed(2),
      PSO: modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.f1_weighted * 100).toFixed(2) : 'N/A',
      GA: 'N/A',
      Bayesian: 'N/A',
    },
    {
      metric: 'Precision',
      Baseline: (modelInfo.baseline_metrics.precision * 100).toFixed(2),
      PSO: modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.precision * 100).toFixed(2) : 'N/A',
      GA: 'N/A',
      Bayesian: 'N/A',
    },
    {
      metric: 'Recall',
      Baseline: (modelInfo.baseline_metrics.recall * 100).toFixed(2),
      PSO: modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.recall * 100).toFixed(2) : 'N/A',
      GA: 'N/A',
      Bayesian: 'N/A',
    },
  ]

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Comparison</CardTitle>
          <CardDescription>
            Comparison between baseline and optimized models (PSO, GA, Bayesian)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Award className="h-5 w-5 text-blue-600" />
                <p className="text-sm font-medium text-blue-900">Accuracy Improvement</p>
              </div>
              <p className="text-2xl font-bold text-blue-600">
                {modelInfo.improvement_percentage?.accuracy 
                  ? `+${modelInfo.improvement_percentage.accuracy.toFixed(2)}%`
                  : 'N/A'}
              </p>
            </div>

            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="h-5 w-5 text-green-600" />
                <p className="text-sm font-medium text-green-900">F1 Score Improvement</p>
              </div>
              <p className="text-2xl font-bold text-green-600">
                {modelInfo.improvement_percentage?.f1_weighted 
                  ? `+${modelInfo.improvement_percentage.f1_weighted.toFixed(2)}%`
                  : 'N/A'}
              </p>
            </div>

            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="h-5 w-5 text-purple-600" />
                <p className="text-sm font-medium text-purple-900">Optimization Method</p>
              </div>
              <p className="text-2xl font-bold text-purple-600">PSO / GA / Bayesian</p>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="Baseline" fill="#94a3b8" radius={[8, 8, 0, 0]} />
              <Bar dataKey="PSO" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              <Bar dataKey="GA" fill="#10b981" radius={[8, 8, 0, 0]} />
              <Bar dataKey="Bayesian" fill="#f59e0b" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Detailed Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-4">Metric</th>
                  <th className="text-right py-2 px-4">Baseline</th>
                  <th className="text-right py-2 px-4">PSO</th>
                  <th className="text-right py-2 px-4">GA</th>
                  <th className="text-right py-2 px-4">Bayesian</th>
                  <th className="text-right py-2 px-4">Best</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="py-2 px-4 font-medium">Accuracy</td>
                  <td className="text-right py-2 px-4">
                    {(modelInfo.baseline_metrics.accuracy * 100).toFixed(2)}%
                  </td>
                  <td className="text-right py-2 px-4">
                    {modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                  </td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4 text-green-600 font-medium">
                    {modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 px-4 font-medium">F1 Score (Weighted)</td>
                  <td className="text-right py-2 px-4">
                    {(modelInfo.baseline_metrics.f1_weighted * 100).toFixed(2)}%
                  </td>
                  <td className="text-right py-2 px-4">
                    {modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.f1_weighted * 100).toFixed(2) + '%' : 'N/A'}
                  </td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4 text-green-600 font-medium">
                    {modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.f1_weighted * 100).toFixed(2) + '%' : 'N/A'}
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 px-4 font-medium">F1 Score (Macro)</td>
                  <td className="text-right py-2 px-4">
                    {(modelInfo.baseline_metrics.f1_macro * 100).toFixed(2)}%
                  </td>
                  <td className="text-right py-2 px-4">
                    {modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.f1_macro * 100).toFixed(2) + '%' : 'N/A'}
                  </td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4 text-green-600 font-medium">
                    {modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.f1_macro * 100).toFixed(2) + '%' : 'N/A'}
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 px-4 font-medium">Precision</td>
                  <td className="text-right py-2 px-4">
                    {(modelInfo.baseline_metrics.precision * 100).toFixed(2)}%
                  </td>
                  <td className="text-right py-2 px-4">
                    {modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.precision * 100).toFixed(2) + '%' : 'N/A'}
                  </td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4">-</td>
                </tr>
                <tr>
                  <td className="py-2 px-4 font-medium">Recall</td>
                  <td className="text-right py-2 px-4">
                    {(modelInfo.baseline_metrics.recall * 100).toFixed(2)}%
                  </td>
                  <td className="text-right py-2 px-4">
                    {modelInfo.optimized_metrics ? (modelInfo.optimized_metrics.recall * 100).toFixed(2) + '%' : 'N/A'}
                  </td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4">N/A</td>
                  <td className="text-right py-2 px-4">-</td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
