'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/app/components/ui/card'
import { Button } from '@/app/components/ui/button'
import { Maximize2, Play, Pause, Square } from 'lucide-react'
import {
  LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell, ComposedChart, Area
} from 'recharts'

interface PSOVisualizationProps {
  psoHistory: any[]
  psoAnimation: any[]
}

export default function PSOVisualization({ psoHistory, psoAnimation }: PSOVisualizationProps) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])
  const [selectedIteration, setSelectedIteration] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [fullscreenChart, setFullscreenChart] = useState<string | null>(null)
  const [playbackSpeed, setPlaybackSpeed] = useState(1000) // ms per iteration

  // Extract unique iterations
  const iterations = Array.from(new Set(psoAnimation.map((p: any) => p.iteration))).sort((a: number, b: number) => a - b)
  const maxIteration = Math.max(...iterations, 0)

  // Get particles for selected iteration
  const getCurrentIterationParticles = () => {
    return psoAnimation.filter(p => p.iteration === selectedIteration)
  }

  // Get parameter evolution for a specific particle
  const getParticleEvolution = (particleId: number) => {
    return psoAnimation
      .filter(p => p.particle_id === particleId)
      .sort((a, b) => a.iteration - b.iteration)
  }

  // Get all particles evolution for a specific parameter
  const getParameterEvolution = (paramName: string) => {
    const data: any[] = []
    iterations.forEach(iter => {
      const particles = psoAnimation.filter(p => p.iteration === iter)
      particles.forEach((p, idx) => {
        data.push({
          iteration: iter,
          particle: idx,
          value: p[paramName],
          is_best: p.is_best,
          f1_score: p.f1_score
        })
      })
    })
    return data
  }

  // Play animation
  const playAnimation = () => {
    setIsPlaying(true)
    let current = selectedIteration
    const interval = setInterval(() => {
      current++
      if (current > maxIteration) {
        clearInterval(interval)
        setIsPlaying(false)
        return
      }
      setSelectedIteration(current)
    }, playbackSpeed)

    // Store interval ID to allow stopping
    return interval
  }

  // Auto-play effect
  useEffect(() => {
    if (isPlaying) {
      const interval = playAnimation()
      return () => clearInterval(interval)
    }
  }, [isPlaying, selectedIteration, maxIteration, playbackSpeed])

  const ChartFullscreen = ({ title, children }: { title: string; children: React.ReactNode }) => {
    if (fullscreenChart !== title) return null
    
    return (
      <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg w-full h-full max-w-7xl max-h-[95vh] p-6 overflow-auto">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold">{title}</h2>
            <Button onClick={() => setFullscreenChart(null)} variant="outline">
              Close
            </Button>
          </div>
          <div className="h-[calc(100%-4rem)]">
            {children}
          </div>
        </div>
      </div>
    )
  }

  if (!mounted) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>PSO Visualization</CardTitle>
          <CardDescription>Loading…</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  if (psoAnimation.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>PSO Visualization</CardTitle>
          <CardDescription>No PSO data available. Run PSO optimization to see visualizations.</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Animation Controls */}
      <Card>
        <CardHeader>
          <CardTitle>PSO Animation Controls</CardTitle>
          <CardDescription>
            Watch how particles explore the hyperparameter space across {maxIteration + 1} iterations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="flex gap-2">
                <Button
                  onClick={() => {
                    if (isPlaying) {
                      setIsPlaying(false)
                    } else {
                      setIsPlaying(true)
                    }
                  }}
                  size="sm"
                  className={isPlaying ? 'bg-orange-600 hover:bg-orange-700' : 'bg-green-600 hover:bg-green-700'}
                >
                  {isPlaying ? (
                    <>
                      <Pause className="mr-2 h-4 w-4" />
                      Pause
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Play
                    </>
                  )}
                </Button>
                {isPlaying && (
                  <Button
                    onClick={() => {
                      setIsPlaying(false)
                      setSelectedIteration(0)
                    }}
                    size="sm"
                    variant="outline"
                  >
                    <Square className="mr-2 h-4 w-4" />
                    Stop
                  </Button>
                )}
              </div>
              
              <div className="flex-1">
                <label className="text-sm font-medium mb-2 block">
                  Iteration: {selectedIteration} / {maxIteration}
                </label>
                <input
                  type="range"
                  min="0"
                  max={maxIteration}
                  value={selectedIteration}
                  onChange={(e) => setSelectedIteration(parseInt(e.target.value))}
                  className="w-full"
                  disabled={isPlaying}
                />
              </div>
              
              <div>
                <label className="text-sm font-medium mb-2 block">Speed (ms)</label>
                <select
                  value={playbackSpeed}
                  onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
                  className="border rounded p-2"
                  disabled={isPlaying}
                >
                  <option value="500">Fast (500ms)</option>
                  <option value="1000">Normal (1s)</option>
                  <option value="2000">Slow (2s)</option>
                </select>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Swarm Movement Visualization */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>🐝 Swarm Movement in Hyperparameter Space</CardTitle>
              <CardDescription>
                Watch particles explore Learning Rate vs Dropout • Iteration {selectedIteration}/{maxIteration}
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setFullscreenChart('swarm_movement')}
            >
              <Maximize2 className="h-4 w-4 mr-2" />
              Fullscreen
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={900}>
            <ComposedChart margin={{ top: 30, right: 50, bottom: 80, left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d1d5db" strokeWidth={1.5} />
              <XAxis 
                type="number" 
                dataKey="learning_rate" 
                name="Learning Rate"
                domain={[0, 0.0005]}
                label={{ value: 'Learning Rate', position: 'insideBottom', offset: -20, style: { fontSize: 18, fontWeight: 'bold' } }}
                tick={{ fontSize: 14 }}
                tickFormatter={(value) => value.toExponential(2)}
              />
              <YAxis 
                type="number" 
                dataKey="dropout" 
                name="Dropout"
                domain={[0, 0.5]}
                label={{ value: 'Dropout', angle: -90, position: 'insideLeft', style: { fontSize: 18, fontWeight: 'bold' } }}
                tick={{ fontSize: 14 }}
                tickFormatter={(value) => value.toFixed(2)}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="bg-white p-4 border-2 border-purple-400 rounded-lg shadow-2xl">
                        <p className="font-bold text-lg text-purple-900 mb-2">Particle {data.particle_id}</p>
                        <div className="space-y-1">
                          <p className="text-sm"><span className="font-semibold">Learning Rate:</span> {data.learning_rate?.toExponential(4)}</p>
                          <p className="text-sm"><span className="font-semibold">Dropout:</span> {data.dropout?.toFixed(3)}</p>
                          <p className="text-sm"><span className="font-semibold">Batch Size:</span> {data.batch_size}</p>
                          <p className="text-sm"><span className="font-semibold">Frozen:</span> {data.frozen_layers}</p>
                          <p className="text-base font-bold text-purple-700 mt-2">F1: {(data.f1_score * 100).toFixed(2)}%</p>
                          {data.is_best && <p className="text-sm text-green-600 font-bold mt-2 flex items-center gap-1">⭐ BEST PARTICLE</p>}
                        </div>
                      </div>
                    )
                  }
                  return null
                }}
              />
              
              {/* Show trails for last few iterations */}
              {selectedIteration > 0 && [...Array(Math.min(3, selectedIteration))].map((_, trailIdx) => {
                const trailIter = selectedIteration - trailIdx - 1
                const trailParticles = psoAnimation.filter(p => p.iteration === trailIter)
                return (
                  <Scatter 
                    key={`trail-${trailIter}`}
                    data={trailParticles}
                    fill="#cbd5e1"
                    opacity={0.4 - trailIdx * 0.12}
                  >
                    {trailParticles.map((entry, index) => (
                      <Cell 
                        key={`trail-cell-${index}`} 
                        fill={`rgba(100, 116, 139, ${0.4 - trailIdx * 0.12})`}
                        r={8}
                      />
                    ))}
                  </Scatter>
                )
              })}
              
              {/* Current iteration particles - MUCH BIGGER */}
              <Scatter 
                data={getCurrentIterationParticles()}
                fill="#8884d8"
                isAnimationActive={false}
              >
                {getCurrentIterationParticles().map((entry, index) => {
                  // Calculate vibrant color based on F1 score
                  const f1 = entry.f1_score || 0
                  let color
                  if (entry.is_best) {
                    color = '#10b981' // Bright green for best
                  } else if (f1 < 0.3) {
                    color = '#ef4444' // Bright red for low F1
                  } else if (f1 < 0.5) {
                    color = '#f97316' // Orange for medium-low F1
                  } else if (f1 < 0.65) {
                    color = '#eab308' // Yellow for medium F1
                  } else {
                    color = '#3b82f6' // Bright blue for high F1
                  }
                  
                  return (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={color}
                      r={entry.is_best ? 28 : 20}
                      stroke={entry.is_best ? '#047857' : '#1f2937'}
                      strokeWidth={entry.is_best ? 6 : 4}
                    />
                  )
                })}
              </Scatter>
            </ComposedChart>
          </ResponsiveContainer>
          <div className="mt-4 space-y-3">
            <div className="flex items-center justify-center gap-6 text-sm bg-gray-50 p-4 rounded-lg flex-wrap">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-green-500 border-4 border-green-800"></div>
                <span className="font-bold">Best Particle</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-blue-500 border-2 border-gray-800"></div>
                <span className="font-semibold">High F1 (≥0.65)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-yellow-500 border-2 border-gray-800"></div>
                <span className="font-semibold">Medium F1 (0.5-0.65)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-orange-500 border-2 border-gray-800"></div>
                <span className="font-semibold">Low F1 (0.3-0.5)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-red-500 border-2 border-gray-800"></div>
                <span className="font-semibold">Very Low F1 (&lt;0.3)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-gray-400 opacity-50"></div>
                <span className="font-medium">Trail (Previous)</span>
              </div>
              <div className="text-xs text-center text-gray-500 mt-2">
                <p>X-axis: Learning Rate (log scale)</p>
                <p>Y-axis: Dropout Rate</p>
              </div>
            </div>
            <div className="text-center text-base text-gray-700 bg-blue-50 p-3 rounded-lg border-2 border-blue-200">
              💡 <span className="font-bold">Tip:</span> Watch how particles explore the learning rate and dropout space. The best configurations will have higher F1 scores (darker blue).
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Parameter Evolution Summary */}
      <Card>
        <CardHeader>
          <CardTitle>📊 Parameter Evolution Summary</CardTitle>
          <CardDescription>Average, Min, and Max values across all particles per iteration</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            {/* Learning Rate Summary */}
            <div>
              <h4 className="font-semibold mb-2 text-sm">Learning Rate</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={iterations.map(iter => {
                  const particles = psoAnimation.filter(p => p.iteration === iter)
                  const lrs = particles.map(p => p.learning_rate).filter(v => v)
                  return {
                    iteration: iter,
                    avg: lrs.reduce((a, b) => a + b, 0) / lrs.length,
                    min: Math.min(...lrs),
                    max: Math.max(...lrs)
                  }
                })}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="avg" stroke="#8b5cf6" strokeWidth={2} name="Average" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="min" stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" name="Min" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={1} strokeDasharray="3 3" name="Max" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Batch Size Summary */}
            <div>
              <h4 className="font-semibold mb-2 text-sm">Batch Size</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={iterations.map(iter => {
                  const particles = psoAnimation.filter(p => p.iteration === iter)
                  const bs = particles.map(p => p.batch_size).filter(v => v)
                  return {
                    iteration: iter,
                    avg: bs.reduce((a, b) => a + b, 0) / bs.length,
                    min: Math.min(...bs),
                    max: Math.max(...bs)
                  }
                })}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="avg" stroke="#10b981" strokeWidth={2} name="Average" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="min" stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" name="Min" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={1} strokeDasharray="3 3" name="Max" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Dropout Summary */}
            <div>
              <h4 className="font-semibold mb-2 text-sm">Dropout</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={iterations.map(iter => {
                  const particles = psoAnimation.filter(p => p.iteration === iter)
                  const drops = particles.map(p => p.dropout).filter(v => v !== undefined)
                  return {
                    iteration: iter,
                    avg: drops.reduce((a, b) => a + b, 0) / drops.length,
                    min: Math.min(...drops),
                    max: Math.max(...drops)
                  }
                })}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="avg" stroke="#f59e0b" strokeWidth={2} name="Average" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="min" stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" name="Min" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={1} strokeDasharray="3 3" name="Max" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Frozen Layers Summary */}
            <div>
              <h4 className="font-semibold mb-2 text-sm">Frozen Layers</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={iterations.map(iter => {
                  const particles = psoAnimation.filter(p => p.iteration === iter)
                  const frozen = particles.map(p => p.frozen_layers).filter(v => v !== undefined)
                  return {
                    iteration: iter,
                    avg: frozen.reduce((a, b) => a + b, 0) / frozen.length,
                    min: Math.min(...frozen),
                    max: Math.max(...frozen)
                  }
                })}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="avg" stroke="#ec4899" strokeWidth={2} name="Average" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="min" stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" name="Min" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={1} strokeDasharray="3 3" name="Max" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="mt-4 flex items-center justify-center gap-6 text-xs bg-gray-50 p-2 rounded">
            <div className="flex items-center gap-1">
              <div className="w-8 h-0.5 bg-purple-600"></div>
              <span>Average</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-8 h-0.5 bg-blue-600 border-dashed border-t-2"></div>
              <span>Min</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-8 h-0.5 bg-red-600 border-dashed border-t-2"></div>
              <span>Max</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* F1 Score Evolution */}
      <Card>
        <CardHeader>
          <CardTitle>🎯 F1 Score Evolution & Convergence</CardTitle>
          <CardDescription>
            Best F1 score found per iteration and swarm performance distribution
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4">
            {/* Best F1 Score per Iteration */}
            <div>
              <h4 className="font-semibold mb-2 text-sm">Best F1 Score Progress</h4>
              <ResponsiveContainer width="100%" height={250}>
                <ComposedChart data={iterations.map(iter => {
                  const particles = psoAnimation.filter(p => p.iteration === iter)
                  const f1s = particles.map(p => p.f1_score).filter(v => v)
                  const best = Math.max(...f1s)
                  const avg = f1s.reduce((a, b) => a + b, 0) / f1s.length
                  return {
                    iteration: iter,
                    best: best,
                    avg: avg,
                    min: Math.min(...f1s)
                  }
                })}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'F1 Score', angle: -90, position: 'insideLeft' }} domain={[0, 1]} />
                  <Tooltip />
                  <Area isAnimationActive={false} animationDuration={0} type="monotone" dataKey="min" fill="#fecaca" stroke="none" fillOpacity={0.3} name="Min" />
                  <Area isAnimationActive={false} animationDuration={0} type="monotone" dataKey="avg" fill="#bfdbfe" stroke="none" fillOpacity={0.4} name="Avg Range" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="best" stroke="#10b981" strokeWidth={3} dot={{ r: 5, fill: '#10b981' }} name="Best F1" />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="avg" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" name="Average F1" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Swarm Diversity */}
            <div>
              <h4 className="font-semibold mb-2 text-sm">Swarm Diversity (Parameter Spread)</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={iterations.map(iter => {
                  const particles = psoAnimation.filter(p => p.iteration === iter)
                  const lrs = particles.map(p => p.learning_rate).filter(v => v)
                  const bs = particles.map(p => p.batch_size).filter(v => v)
                  const drops = particles.map(p => p.dropout).filter(v => v !== undefined)
                  
                  // Calculate standard deviation as measure of diversity
                  const lrStd = Math.sqrt(lrs.reduce((sum, v) => sum + Math.pow(v - lrs.reduce((a,b) => a+b, 0)/lrs.length, 2), 0) / lrs.length)
                  const bsStd = Math.sqrt(bs.reduce((sum, v) => sum + Math.pow(v - bs.reduce((a,b) => a+b, 0)/bs.length, 2), 0) / bs.length)
                  const dropStd = Math.sqrt(drops.reduce((sum, v) => sum + Math.pow(v - drops.reduce((a,b) => a+b, 0)/drops.length, 2), 0) / drops.length)
                  
                  return {
                    iteration: iter,
                    diversity: (lrStd / Math.max(...lrs) + bsStd / Math.max(...bs) + dropStd / Math.max(...drops)) / 3
                  }
                })}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} label={{ value: 'Diversity', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                  <Tooltip />
                  <Line isAnimationActive={false} animationDuration={0} type="monotone" dataKey="diversity" stroke="#f59e0b" strokeWidth={2} name="Swarm Diversity" />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-xs text-gray-600 mt-2 text-center">
                Higher diversity = particles exploring widely • Lower diversity = particles converging
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Fullscreen Modal for Swarm Movement */}
      <ChartFullscreen title="swarm_movement">
        <div className="h-full flex flex-col">
          <h2 className="text-2xl font-bold mb-4">🐝 Swarm Movement - Iteration {selectedIteration}/{maxIteration}</h2>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart margin={{ top: 20, right: 40, bottom: 80, left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                type="number" 
                dataKey="learning_rate" 
                name="Learning Rate"
                domain={['dataMin', 'dataMax']}
                label={{ value: 'Learning Rate', position: 'insideBottom', offset: -25, style: { fontSize: 18, fontWeight: 'bold' } }}
                tick={{ fontSize: 14 }}
              />
              <YAxis 
                type="number" 
                dataKey="dropout" 
                name="Dropout"
                domain={[0, 0.5]}
                label={{ value: 'Dropout', angle: -90, position: 'insideLeft', style: { fontSize: 18, fontWeight: 'bold' } }}
                tick={{ fontSize: 14 }}
                tickFormatter={(value) => (value as number).toFixed(2)}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="bg-white p-4 border-2 border-purple-400 rounded-lg shadow-2xl">
                        <p className="font-bold text-xl text-purple-900 mb-2">Particle {data.particle_id}</p>
                        <div className="space-y-1">
                          <p className="text-base"><span className="font-semibold">LR:</span> {data.learning_rate?.toExponential(3)}</p>
                          <p className="text-base"><span className="font-semibold">Batch:</span> {data.batch_size}</p>
                          <p className="text-base"><span className="font-semibold">Dropout:</span> {data.dropout?.toFixed(3)}</p>
                          <p className="text-base"><span className="font-semibold">Frozen:</span> {data.frozen_layers}</p>
                          <p className="text-lg font-bold text-purple-700 mt-2">F1: {(data.f1_score * 100).toFixed(2)}%</p>
                          {data.is_best && <p className="text-base text-green-600 font-bold mt-2">⭐ BEST PARTICLE</p>}
                        </div>
                      </div>
                    )
                  }
                  return null
                }}
              />
              
              {/* Show trails */}
              {selectedIteration > 0 && [...Array(Math.min(3, selectedIteration))].map((_, trailIdx) => {
                const trailIter = selectedIteration - trailIdx - 1
                const trailParticles = psoAnimation.filter(p => p.iteration === trailIter)
                return (
                  <Scatter 
                    key={`trail-${trailIter}`}
                    data={trailParticles}
                    fill="#cbd5e1"
                    isAnimationActive={false}
                  >
                    {trailParticles.map((entry, index) => (
                      <Cell 
                        key={`trail-cell-${index}`} 
                        fill={`rgba(100, 116, 139, ${0.4 - trailIdx * 0.12})`}
                        r={8}
                      />
                    ))}
                  </Scatter>
                )
              })}
              
              {/* Current particles */}
              <Scatter 
                data={getCurrentIterationParticles()}
                fill="#8884d8"
              >
                {getCurrentIterationParticles().map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.is_best ? '#10b981' : `hsl(${(entry.f1_score || 0) * 240}, 85%, 50%)`}
                    r={entry.is_best ? 24 : 18}
                    stroke={entry.is_best ? '#059669' : '#ffffff'}
                    strokeWidth={entry.is_best ? 6 : 4}
                  />
                ))}
              </Scatter>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </ChartFullscreen>

    </div>
  )
}
