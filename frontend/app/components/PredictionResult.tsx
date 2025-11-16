'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/app/components/ui/card'
import { PredictionResponse } from '@/lib/api'
import { getLabelColor, getLabelIcon, formatPercentage } from '@/lib/utils'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface PredictionResultProps {
  result: PredictionResponse
}

export default function PredictionResult({ result }: PredictionResultProps) {
  const chartData = Object.entries(result.all_probabilities).map(([label, probability]) => ({
    label,
    probability: probability * 100,
  }))

  const getBarColor = (label: string) => {
    const colors: Record<string, string> = {
      'hate': '#dc2626',
      'not-hate': '#16a34a',
      '1': '#dc2626',
      '0': '#16a34a',
    }
    return colors[label] || '#6b7280'
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Analysis Result</CardTitle>
          <CardDescription>Model: {result.model_used}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground mb-2">Statement:</p>
              <p className="text-base">{result.text}</p>
            </div>

            <div className={`p-6 rounded-lg border-2 ${getLabelColor(result.predicted_label)}`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium mb-1">Predicted Label</p>
                  <p className="text-3xl font-bold flex items-center gap-2">
                    <span>{getLabelIcon(result.predicted_label)}</span>
                    <span className="capitalize">{result.predicted_label.replace('-', ' ')}</span>
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium mb-1">Confidence</p>
                  <p className="text-3xl font-bold">{formatPercentage(result.confidence)}</p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Probability Distribution</CardTitle>
          <CardDescription>Confidence scores for all labels</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="label" 
                angle={-45}
                textAnchor="end"
                height={100}
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value: number) => `${value.toFixed(2)}%`}
                labelFormatter={(label) => label.toString().replace('-', ' ')}
              />
              <Bar dataKey="probability" radius={[8, 8, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getBarColor(entry.label)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-4 space-y-2">
            {Object.entries(result.all_probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([label, probability]) => (
                <div key={label} className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <span>{getLabelIcon(label)}</span>
                    <span className="capitalize">{label.replace('-', ' ')}</span>
                  </span>
                  <span className="font-medium">{formatPercentage(probability)}</span>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
