'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card'
import { Button } from '@/app/components/ui/button'
import type { OptimizationConfig } from '@/lib/api'

interface ParameterOptimizationControlsProps {
  algorithm: 'pso' | 'ga' | 'bayesian'
  defaultParams: {
    learning_rate: number
    batch_size: number
    dropout: number
    frozen_layers: number
  }
  initialConfig?: OptimizationConfig
  onSave: (params: {
    optimize: {
      learning_rate?: boolean
      batch_size?: boolean
      dropout?: boolean
      frozen_layers?: boolean
    }
    fixed: {
      learning_rate?: number
      batch_size?: number
      dropout?: number
      frozen_layers?: number
    }
  }) => void
  isTraining: boolean
}

export function ParameterOptimizationControls({
  algorithm,
  defaultParams,
  initialConfig,
  onSave,
  isTraining,
}: ParameterOptimizationControlsProps) {
  const [optimizeParams, setOptimizeParams] = useState({
    learning_rate: initialConfig?.optimize?.learning_rate ?? true,
    batch_size: initialConfig?.optimize?.batch_size ?? true,
    dropout: initialConfig?.optimize?.dropout ?? true,
    frozen_layers: initialConfig?.optimize?.frozen_layers ?? true,
  })

  const [fixedValues, setFixedValues] = useState({
    learning_rate: initialConfig?.fixed?.learning_rate ?? defaultParams.learning_rate,
    batch_size: initialConfig?.fixed?.batch_size ?? defaultParams.batch_size,
    dropout: initialConfig?.fixed?.dropout ?? defaultParams.dropout,
    frozen_layers: initialConfig?.fixed?.frozen_layers ?? defaultParams.frozen_layers,
  })

  useEffect(() => {
    if (!initialConfig) return
    setOptimizeParams({
      learning_rate: initialConfig.optimize?.learning_rate ?? true,
      batch_size: initialConfig.optimize?.batch_size ?? true,
      dropout: initialConfig.optimize?.dropout ?? true,
      frozen_layers: initialConfig.optimize?.frozen_layers ?? true,
    })
    setFixedValues({
      learning_rate: initialConfig.fixed?.learning_rate ?? defaultParams.learning_rate,
      batch_size: initialConfig.fixed?.batch_size ?? defaultParams.batch_size,
      dropout: initialConfig.fixed?.dropout ?? defaultParams.dropout,
      frozen_layers: initialConfig.fixed?.frozen_layers ?? defaultParams.frozen_layers,
    })
  }, [initialConfig, defaultParams.learning_rate, defaultParams.batch_size, defaultParams.dropout, defaultParams.frozen_layers])

  const handleToggleOptimize = (param: keyof typeof optimizeParams) => {
    setOptimizeParams(prev => ({
      ...prev,
      [param]: !prev[param],
    }))
  }

  const handleFixedValueChange = (param: keyof typeof fixedValues, value: number) => {
    setFixedValues(prev => ({
      ...prev,
      [param]: value,
    }))
  }

  const handleStart = () => {
    const optimize: Record<string, boolean> = {}
    const fixed: Record<string, number> = {}

    Object.entries(optimizeParams).forEach(([key, value]) => {
      optimize[key] = !!value
      if (!value) {
        fixed[key] = fixedValues[key as keyof typeof fixedValues]
      }
    })

    onSave({ optimize, fixed } as any)
  }

  const paramConfigs = [
    {
      id: 'learning_rate',
      label: 'Learning Rate',
      description: 'Step size at each iteration while moving toward a minimum of the loss function',
      min: 1e-6,
      max: 1e-3,
      step: 1e-6,
      format: (v: number) => v.toExponential(2),
    },
    {
      id: 'batch_size',
      label: 'Batch Size',
      description: 'Number of training examples utilized in one iteration',
      min: 4,
      max: 64,
      step: 1,
      format: (v: number) => Math.round(v).toString(),
    },
    {
      id: 'dropout',
      label: 'Dropout Rate',
      description: 'Fraction of input units to drop during training to prevent overfitting',
      min: 0,
      max: 0.5,
      step: 0.01,
      format: (v: number) => v.toFixed(2),
    },
    {
      id: 'frozen_layers',
      label: 'Frozen Layers',
      description: 'Number of transformer layers to freeze during training (0 = fine-tune all)',
      min: 0,
      max: 6,
      step: 1,
      format: (v: number) => Math.round(v).toString(),
    },
  ]

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Optimization Parameters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          {paramConfigs.map((param) => (
            <div key={param.id} className="space-y-2">
              <div className="flex items-center justify-between">
                <label htmlFor={`optimize-${param.id}`} className="font-medium">
                  {param.label}
                </label>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-muted-foreground">
                    {optimizeParams[param.id as keyof typeof optimizeParams] ? 'Optimizing' : 'Fixed'}
                  </span>
                  <input
                    id={`optimize-${param.id}`}
                    type="checkbox"
                    checked={optimizeParams[param.id as keyof typeof optimizeParams]}
                    onChange={() => handleToggleOptimize(param.id as any)}
                    disabled={isTraining}
                  />
                </div>
              </div>
              
              {!optimizeParams[param.id as keyof typeof optimizeParams] && (
                <div className="pl-2">
                  <div className="flex items-center gap-4">
                    <input
                      id={`fixed-${param.id}`}
                      type="range"
                      min={param.min}
                      max={param.max}
                      step={param.step}
                      value={fixedValues[param.id as keyof typeof fixedValues] as number}
                      onChange={(e) => handleFixedValueChange(param.id as any, Number(e.target.value))}
                      disabled={isTraining}
                      className="flex-1"
                    />
                    <input
                      type="number"
                      value={fixedValues[param.id as keyof typeof fixedValues] as number}
                      onChange={(e) => handleFixedValueChange(param.id as any, Number(e.target.value))}
                      min={param.min}
                      max={param.max}
                      step={param.step}
                      className="w-24 border p-1 rounded"
                      disabled={isTraining}
                    />
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Range: {param.format(param.min)} to {param.format(param.max)}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        <Button
          onClick={handleStart}
          disabled={isTraining}
          className="w-full"
        >
          {isTraining ? 'Optimizing...' : 'Save Settings'}
        </Button>

        {Object.values(optimizeParams).every(v => !v) && (
          <p className="text-sm text-destructive text-center">
            Please select at least one parameter to optimize
          </p>
        )}
      </CardContent>
    </Card>
  )
}
