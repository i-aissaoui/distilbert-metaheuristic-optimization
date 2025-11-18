'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/app/components/ui/card'
import { Button } from '@/app/components/ui/button'
import { api, TrainingParams, PSOParams, GAParams, BayesianParams, OptimizationConfig } from '@/lib/api'
import { Loader2, Play, Settings, TrendingUp, Zap, BarChart3, Maximize2, CheckCircle, CheckCircle2, Trophy } from 'lucide-react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell
} from 'recharts'
import PSOVisualization from '@/app/components/PSOVisualization'
import { ParameterOptimizationControls } from './ParameterOptimizationControls'

interface TrainingProps {
  mode: 'optimization'
  onTrainingComplete?: () => void
}

export default function Training({ mode, onTrainingComplete }: TrainingProps) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])
  const [activeMode, setActiveMode] = useState<'pso' | 'ga' | 'bayesian'>('pso')
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [psoProgress, setPsoProgress] = useState(0)
  const [currentParticleMetrics, setCurrentParticleMetrics] = useState<any>(null)
  
  // Full data training state
  const [fullDataTraining, setFullDataTraining] = useState<{
    active: boolean
    algorithm: string
    progress: number
    message: string
  } | null>(null)
  
  // Epochs for full data training
  const [psoEpochs, setPsoEpochs] = useState(3)
  const [gaEpochs, setGaEpochs] = useState(3)
  const [bayesianEpochs, setBayesianEpochs] = useState(3)
  
  // Custom training parameters
  const [customParams, setCustomParams] = useState<TrainingParams>({
    learning_rate: 2e-5,
    batch_size: 16,
    dropout: 0.1,
    frozen_layers: 0,
    epochs: 1
  })
  
  // PSO parameters (optimized defaults)
  const [psoParams, setPsoParams] = useState<PSOParams>({
    swarmsize: 10,
    maxiter: 15
  })
  
  // GA parameters (optimized defaults)
  const [gaParams, setGaParams] = useState<GAParams>({
    population_size: 10,
    num_generations: 15
  })
  
  // Bayesian parameters (optimized defaults)
  const [bayesianParams, setBayesianParams] = useState<BayesianParams>({
    n_trials: 20
  })
  
  // Results data
  const [customResults, setCustomResults] = useState<any>(null)
  const [psoResults, setPsoResults] = useState<any>(null)
  const [gaResults, setGaResults] = useState<any>(null)
  const [bayesianResults, setBayesianResults] = useState<any>(null)
  const [comparisonData, setComparisonData] = useState<any[]>([])
  const [psoHistory, setPsoHistory] = useState<any[]>([])
  const [psoAnimation, setPsoAnimation] = useState<any[]>([])
  const [gaHistory, setGaHistory] = useState<any[]>([])
  const [gaAnimation, setGaAnimation] = useState<any[]>([])
  const [bayesianHistory, setBayesianHistory] = useState<any[]>([])
  const [bayesianAnimation, setBayesianAnimation] = useState<any[]>([])
  const [currentIteration, setCurrentIteration] = useState(0)
  const [fullscreenChart, setFullscreenChart] = useState<string | null>(null)
  
  // Full data training results
  const [psoFullDataResults, setPsoFullDataResults] = useState<any>(null)
  const [gaFullDataResults, setGaFullDataResults] = useState<any>(null)
  const [bayesianFullDataResults, setBayesianFullDataResults] = useState<any>(null)
  
  // Hyperparameters data
  const [hyperparameters, setHyperparameters] = useState<any>(null)
  
  // Training logs
  const [trainingLogs, setTrainingLogs] = useState<any[]>([])
  const [lastTrainingTime, setLastTrainingTime] = useState<string | null>(null)
  
  // Track which algorithm is currently training
  const [activeTraining, setActiveTraining] = useState<string | null>(null)

  // Persisted optimization config (optimize/fixed) loaded from backend
  const [optimizationConfig, setOptimizationConfig] = useState<OptimizationConfig | null>(null)
  const [toastMessage, setToastMessage] = useState<string | null>(null)

  useEffect(() => {
    loadHyperparameters()
    // Load saved optimize/fixed config so UI restores after refresh
    api.getOptimizationConfig().then(setOptimizationConfig).catch(() => {})
    // Load saved algorithm settings (PSO/GA/Bayesian) to restore after refresh
    api.getAlgorithmSettings()
      .then((settings) => {
        try {
          if (settings?.pso) setPsoParams((prev) => ({ ...prev, ...settings.pso }))
          if (settings?.ga) setGaParams((prev) => ({ ...prev, ...settings.ga }))
          if (settings?.bayesian) setBayesianParams((prev) => ({ ...prev, ...settings.bayesian }))
        } catch {}
      })
      .catch(() => {})
    loadSavedResults()  // Load all algorithm results
    loadPSOHistory()
    loadGAHistory()
    loadBayesianHistory()
    loadSavedLogs()
    loadSavedResults()
    
    // Poll for results every 3 seconds
    const resultsPollInterval = setInterval(() => {
      loadSavedResults()
    }, 3000)
    
    // Reload results when tab becomes visible
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        loadSavedResults()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    
    return () => {
      clearInterval(resultsPollInterval)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [])
  
  // Poll for full data training progress
  useEffect(() => {
    if (!fullDataTraining?.active) return
    
    const pollProgress = setInterval(async () => {
      try {
        const status = await api.getTrainingStatus()
        if (status.status === 'running' && status.progress !== undefined) {
          setFullDataTraining(prev => prev ? {
            ...prev,
            progress: Math.round(status.progress || 0),
            message: status.message || 'Training in progress...'
          } : null)
        } else if (status.status === 'completed') {
          setFullDataTraining(null)
          loadSavedResults() // Reload results after completion
        }
      } catch (error) {
        console.error('Error polling training status:', error)
      }
    }, 2000) // Poll every 2 seconds
    
    return () => clearInterval(pollProgress)
  }, [fullDataTraining?.active])
  
  const loadSavedResults = async () => {
    try {
      // Load saved results from backend JSON files
      const [customResult, psoResult, gaResult, bayesianResult] = await Promise.all([
        api.loadTrainingResult('custom').catch(() => null),
        api.loadTrainingResult('pso').catch(() => null),
        api.loadTrainingResult('ga').catch(() => null),
        api.loadTrainingResult('bayesian').catch(() => null)
      ])
      
      // Set results or default to empty/zero state
      setCustomResults(customResult || { accuracy: 0, f1_score: 0, message: 'No training completed yet' })
      setPsoResults(psoResult || { accuracy: 0, f1_score: 0, message: 'No PSO run yet' })
      setGaResults(gaResult || { accuracy: 0, f1_score: 0, message: 'No GA run yet' })
      setBayesianResults(bayesianResult || { accuracy: 0, f1_score: 0, message: 'No Bayesian run yet' })
      
      // Load full data training results
      loadFullDataResults()
    } catch (error) {
      console.error('Failed to load saved results:', error)
    }
  }
  
  const loadFullDataResults = async () => {
    try {
      // Load full data training results for each algorithm
      const [psoFull, gaFull, bayesianFull] = await Promise.all([
        fetch('http://localhost:8000/load-result/pso/full_data').then(r => r.ok ? r.json() : null).catch(() => null),
        fetch('http://localhost:8000/load-result/ga/full_data').then(r => r.ok ? r.json() : null).catch(() => null),
        fetch('http://localhost:8000/load-result/bayesian/full_data').then(r => r.ok ? r.json() : null).catch(() => null)
      ])
      
      setPsoFullDataResults(psoFull)
      setGaFullDataResults(gaFull)
      setBayesianFullDataResults(bayesianFull)
    } catch (error) {
      console.error('Failed to load full data results:', error)
    }
  }
  
  const loadSavedLogs = () => {
    // No longer loading logs - we only keep last result per algorithm
  }
  
  const saveTrainingLog = (logEntry: any) => {
    // No longer saving logs - we only keep last result per algorithm
  }
  
  const clearTrainingLogs = async () => {
    try {
      // Clear results from backend
      await api.clearTrainingHistory()
      
      // Clear local state
      setTrainingLogs([])
      setLastTrainingTime(null)
      setCustomResults(null)
      setPsoResults(null)
      setGaResults(null)
      setBayesianResults(null)
      
      // No localStorage - everything is in backend
      
      alert('Training history cleared successfully!')
    } catch (error) {
      console.error('Failed to clear history:', error)
      alert('Failed to clear history')
    }
  }

  const loadHyperparameters = async () => {
    try {
      const data = await api.getHyperparameters()
      setHyperparameters(data)
    } catch (error) {
      console.error('Failed to load hyperparameters:', error)
    }
  }

  const getDefaultParamValues = () => {
    const baseline = hyperparameters?.baseline || {
      learning_rate: 2e-5,
      batch_size: 16,
      dropout: 0.1,
      frozen_layers: 0,
    }
    return baseline
  }

  // Derived Bayesian display values (latest metrics with history fallback)
  const lastBayesianHistoryEntry = bayesianHistory.length > 0 ? bayesianHistory[bayesianHistory.length - 1] : null
  const bayesianHistoryParams = lastBayesianHistoryEntry?.best_params
  const bayesianDisplayF1 =
    typeof bayesianResults?.f1_score === 'number' && bayesianResults.f1_score > 0
      ? bayesianResults.f1_score
      : typeof bayesianHistoryParams?.f1_score === 'number'
        ? bayesianHistoryParams.f1_score
        : 0
  const bayesianDisplayAccuracy =
    typeof bayesianResults?.accuracy === 'number' && bayesianResults.accuracy > 0
      ? bayesianResults.accuracy
      : typeof bayesianHistoryParams?.accuracy === 'number'
        ? bayesianHistoryParams.accuracy
        : 0
  const bayesianHasOptimizationMetrics = bayesianDisplayF1 > 0 || bayesianDisplayAccuracy > 0
  const bayesianDisplayParams =
    bayesianResults?.best_params && Object.keys(bayesianResults.best_params).length > 0
      ? bayesianResults.best_params
      : bayesianHistoryParams || {}
  const hasBayesianParams = Object.keys(bayesianDisplayParams).length > 0
  const bayesianFullF1 = typeof bayesianFullDataResults?.f1_score === 'number' ? bayesianFullDataResults.f1_score : null
  const bayesianFullAccuracy = typeof bayesianFullDataResults?.accuracy === 'number' ? bayesianFullDataResults.accuracy : null
  const bayesianMainF1 = bayesianFullF1 ?? bayesianDisplayF1
  const bayesianMainAccuracy = bayesianFullAccuracy ?? bayesianDisplayAccuracy
  const bayesianStatLabelSuffix = bayesianFullDataResults ? ' (Full Data)' : ' (20% data)'
  const bayesianHasMetrics = bayesianHasOptimizationMetrics || bayesianFullF1 !== null || bayesianFullAccuracy !== null

  const handleTabChange = (mode: 'pso' | 'ga' | 'bayesian') => {
    setActiveMode(mode)
  };

  const startWithOptimization = async (
    algorithm: 'pso' | 'ga' | 'bayesian',
    cfg: OptimizationConfig
  ) => {
    try {
      await api.saveOptimizationConfig(cfg)
      if (algorithm === 'pso') {
        await api.saveAlgorithmSettings('pso', psoParams).catch(() => {})
        await startPSOTraining()
      } else if (algorithm === 'ga') {
        await api.saveAlgorithmSettings('ga', gaParams).catch(() => {})
        await startGATraining()
      } else {
        await api.saveAlgorithmSettings('bayesian', bayesianParams).catch(() => {})
        await startBayesianTraining()
      }
    } catch (e) {
      console.error('Failed to start optimization with config:', e)
    }
  }

  const loadPSOHistory = async () => {
    try {
      const history = await api.getPSOHistory()
      if (history.data && history.data.length > 0) {
        setPsoHistory(history.data)
      }
      
      const animation = await api.getPSOAnimation()
      if (animation.data && animation.data.length > 0) {
        setPsoAnimation(animation.data)
      }
    } catch (error) {
      console.error('Failed to load PSO history:', error)
    }
  }

  const loadGAHistory = async () => {
    try {
      const history = await api.getGAHistory()
      if (history.data && history.data.length > 0) {
        setGaHistory(history.data)
      }
      
      const animation = await api.getGAAnimation()
      if (animation.data && animation.data.length > 0) {
        setGaAnimation(animation.data)
      }
    } catch (error) {
      console.error('Failed to load GA history:', error)
    }
  }

  const loadBayesianHistory = async () => {
    try {
      const history = await api.getBayesianHistory()
      if (history.data && history.data.length > 0) {
        setBayesianHistory(history.data)
      }
      
      const animation = await api.getBayesianAnimation()
      if (animation.data && animation.data.length > 0) {
        setBayesianAnimation(animation.data)
      }
    } catch (error) {
      console.error('Failed to load Bayesian history:', error)
    }
  }

  const startCustomTraining = async () => {
    setIsTraining(true)
    setTrainingProgress(0)
    setActiveTraining('Custom Training')
    
    let pollInterval: NodeJS.Timeout | null = null
    let hasCompleted = false
    
    // Start polling immediately
    pollInterval = setInterval(async () => {
      if (hasCompleted) {
        if (pollInterval) clearInterval(pollInterval)
        return
      }
      
      try {
        const status = await api.getTrainingStatus()
        
        if (status.status === 'running' && status.progress !== undefined) {
          setTrainingProgress(status.progress)
          // Update result with current status info
          setCustomResults((prev: any) => ({
            ...prev,
            message: status.message,
            current_epoch: status.current_epoch,
            total_epochs: status.total_epochs,
            loss: status.loss
          }))
        } else if (status.status === 'completed') {
          hasCompleted = true
          if (pollInterval) clearInterval(pollInterval)
          setTrainingProgress(100)
          setIsTraining(false)
          setActiveTraining(null)
          
          // Save results to backend JSON file with final metrics
          const completedResult: any = { 
            type: 'custom',
            completedAt: new Date().toISOString(),
            accuracy: status.accuracy,
            f1_score: status.f1_score,
            f1_weighted: (status as any).f1_weighted,
            f1_macro: (status as any).f1_macro,
            precision: (status as any).precision,
            recall: (status as any).recall,
            loss: status.loss,
            message: status.message,
            training_time: (status as any).training_time,
            parameters: customParams
          }
          setCustomResults(completedResult)
          await api.saveTrainingResult('custom', completedResult)
          
          // Save training log
          saveTrainingLog({
            type: 'custom',
            mode: 'Custom Parameters',
            parameters: customParams,
            result: completedResult,
            status: 'completed'
          })
          
          if (onTrainingComplete) onTrainingComplete()
        }
      } catch (error) {
        console.error('Error polling status:', error)
      }
    }, 2000) // Poll every 2 seconds
    
    // Start training (fire and forget - polling will handle status)
    api.startTraining(customParams).catch((error) => {
      console.error('Failed to start training:', error)
      hasCompleted = true
      if (pollInterval) clearInterval(pollInterval)
      setIsTraining(false)
      setActiveTraining(null)
    })
    
    // Safety timeout - stop polling after 30 minutes
    setTimeout(() => {
      hasCompleted = true
      if (pollInterval) clearInterval(pollInterval)
      if (isTraining) {
        setIsTraining(false)
      }
    }, 1800000)
  }

  const handleCatchError = () => {
    try {
      // Placeholder for error handling
    } catch (error) {
      console.error('Training failed:', error)
      setIsTraining(false)
      
      // Save failed training log
      saveTrainingLog({
        type: 'custom',
        mode: 'Custom Parameters',
        parameters: customParams,
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 'failed'
      })
    }
  }

  const startPSOTraining = async () => {
    setIsTraining(true)
    setPsoProgress(0)
    setCurrentIteration(0)
    setActiveTraining('PSO Optimization')
    
    // Clear full data results when starting new optimization
    setPsoFullDataResults(null)
    
    try {
      const result = await api.startPSO(psoParams)
      setPsoResults(result)
      
      // Poll for real-time updates
      const pollInterval = setInterval(async () => {
        try {
          // Get training status
          const status = await api.getTrainingStatus()
          
          if (status.status === 'running' && status.iteration !== undefined && (status.progress || 0) < 100) {
            // Use actual progress from backend
            setPsoProgress(Math.min(status.progress || 0, 100))
            setCurrentIteration(status.iteration)
            
            // Update current particle metrics if available
            if (status.current_particle_metrics) {
              setCurrentParticleMetrics(status.current_particle_metrics)
            }
            
            // Load latest PSO data
            await loadPSOHistory()
          } else if (status.status === 'completed' || (status.progress || 0) >= 100) {
            // Training is done - either status is completed OR progress is 100%
            setCurrentParticleMetrics(null)
            clearInterval(pollInterval)
            setPsoProgress(100)
            setIsTraining(false)
            setActiveTraining(null)
            
            // Final load of results - RELOAD EVERYTHING
            await loadPSOHistory() // Reloads history AND animation data
            await loadHyperparameters()
            
            // Force refresh animation data
            const animation = await api.getPSOAnimation()
            if (animation.data && animation.data.length > 0) {
              setPsoAnimation(animation.data)
            }
            
            // Load the ACTUAL results from backend latest.json (not from initial result)
            const freshResults = await api.loadTrainingResult('pso')
            if (freshResults) {
              setPsoResults(freshResults)
              
              // Save PSO training log with fresh results
              saveTrainingLog({
                type: 'pso',
                mode: 'PSO Optimization',
                parameters: psoParams,
                result: freshResults,
                status: 'completed'
              })
            }
            
            // Also reload all saved results to update the state
            await loadSavedResults()
            
            if (onTrainingComplete) onTrainingComplete()
          }
        } catch (error) {
          console.error('Error polling status:', error)
        }
      }, 2000) // Poll every 2 seconds
      
      // Safety timeout - stop polling after expected duration
      setTimeout(() => {
        clearInterval(pollInterval)
        if (isTraining) {
          setIsTraining(false)
          loadPSOHistory()
        }
      }, psoParams.maxiter * 60000) // Assume max 1 minute per iteration
      
    } catch (error) {
      console.error('PSO training failed:', error)
      setIsTraining(false)
      
      // Save failed PSO training log
      saveTrainingLog({
        type: 'pso',
        mode: 'PSO Optimization',
        parameters: psoParams,
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 'failed'
      })
    }
  }

  const startGATraining = async () => {
    setIsTraining(true)
    setPsoProgress(0)
    setCurrentIteration(0)
    setActiveTraining('GA Optimization')
    
    // Clear full data results when starting new optimization
    setGaFullDataResults(null)
    
    let pollInterval: NodeJS.Timeout | null = null
    
    try {
      console.log('Starting GA optimization...')
      const result = await api.startGA(gaParams)
      console.log('GA started:', result)
      setGaResults(result)
      
      // Start polling immediately
      pollInterval = setInterval(async () => {
        try {
          const status = await api.getTrainingStatus()
          console.log('GA Status:', status)
          
          if (status.status === 'running' && (status.progress || 0) < 100) {
            const progress = status.progress || ((status.generation || 0) / gaParams.num_generations) * 100
            console.log('GA Progress:', progress)
            setPsoProgress(Math.min(progress, 100))
            setCurrentIteration(status.generation || 0)
            await loadGAHistory()
          } else if (status.status === 'completed' || (status.progress || 0) >= 100) {
            console.log('GA Completed!')
            if (pollInterval) clearInterval(pollInterval)
            setPsoProgress(100)
            setIsTraining(false)
            setActiveTraining(null)
            
            // Final load of results - RELOAD EVERYTHING
            await loadGAHistory() // Reloads history AND animation data
            await loadHyperparameters()
            
            // Force refresh animation data
            const animation = await api.getGAAnimation()
            if (animation.data && animation.data.length > 0) {
              setGaAnimation(animation.data)
            }
            
            // Load the ACTUAL results from backend latest.json (not from initial result)
            const freshResults = await api.loadTrainingResult('ga')
            if (freshResults) {
              setGaResults(freshResults)
              
              // Save GA training log with fresh results
              saveTrainingLog({
                type: 'ga',
                mode: 'Genetic Algorithm',
                parameters: gaParams,
                result: freshResults,
                status: 'completed'
              })
            }
            
            // Also reload all saved results to update the state
            await loadSavedResults()
            
            if (onTrainingComplete) onTrainingComplete()
          }
        } catch (error) {
          console.error('Error polling status:', error)
        }
      }, 2000)
      
      setTimeout(() => {
        if (pollInterval) clearInterval(pollInterval)
        if (isTraining) {
          setIsTraining(false)
          loadGAHistory()
        }
      }, gaParams.num_generations * 60000)
      
    } catch (error) {
      console.error('GA training failed:', error)
      if (pollInterval) clearInterval(pollInterval)
      setIsTraining(false)
      setActiveTraining(null)
      setPsoProgress(0)
      
      saveTrainingLog({
        type: 'ga',
        mode: 'Genetic Algorithm',
        parameters: gaParams,
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 'failed'
      })
    }
  }

  const startBayesianTraining = async () => {
    setIsTraining(true)
    setPsoProgress(0)
    setCurrentIteration(0)
    setActiveTraining('Bayesian Optimization')
    
    // Clear full data results when starting new optimization
    setBayesianFullDataResults(null)
    
    try {
      const result = await api.startBayesian(bayesianParams)
      setBayesianResults(result)
      
      // Poll for real-time updates
      const pollInterval = setInterval(async () => {
        try {
          const status = await api.getTrainingStatus()
          
          if (status.status === 'running' && (status.progress || 0) < 100) {
            const progress = status.progress || ((status.trial || 0) / bayesianParams.n_trials) * 100
            setPsoProgress(Math.min(progress, 100))
            setCurrentIteration(status.trial || 0)
            await loadBayesianHistory()
          } else if (status.status === 'completed' || (status.progress || 0) >= 100) {
            clearInterval(pollInterval)
            setPsoProgress(100)
            setIsTraining(false)
            setActiveTraining(null)
            
            // Final load of results - RELOAD EVERYTHING
            await loadBayesianHistory() // Reloads history AND animation data
            await loadHyperparameters()
            
            // Force refresh animation data
            const animation = await api.getBayesianAnimation()
            if (animation.data && animation.data.length > 0) {
              setBayesianAnimation(animation.data)
            }
            
            // Load the ACTUAL results from backend latest.json (not from initial result)
            const freshResults = await api.loadTrainingResult('bayesian')
            if (freshResults) {
              setBayesianResults(freshResults)
              
              // Save Bayesian training log with fresh results
              saveTrainingLog({
                type: 'bayesian',
                mode: 'Bayesian Optimization',
                parameters: bayesianParams,
                result: freshResults,
                status: 'completed'
              })
            }
            
            // Also reload all saved results to update the state
            await loadSavedResults()
            
            if (onTrainingComplete) onTrainingComplete()
          }
        } catch (error) {
          console.error('Error polling status:', error)
        }
      }, 2000)
      
      setTimeout(() => {
        clearInterval(pollInterval)
        if (isTraining) {
          setIsTraining(false)
          loadBayesianHistory()
        }
      }, bayesianParams.n_trials * 60000)
      
    } catch (error) {
      console.error('Bayesian training failed:', error)
      setIsTraining(false)
      
      saveTrainingLog({
        type: 'bayesian',
        mode: 'Bayesian Optimization',
        parameters: bayesianParams,
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 'failed'
      })
    }
  }

  const generateComparisonData = () => {
    if (!hyperparameters) return []
    
    const baseline = hyperparameters.baseline
    const optimized = hyperparameters.optimized
    
    if (!baseline || !optimized) return []
    
    return [
      {
        parameter: 'Learning Rate',
        Baseline: baseline.learning_rate,
        Optimized: optimized.learning_rate
      },
      {
        parameter: 'Batch Size',
        Baseline: baseline.batch_size,
        Optimized: optimized.batch_size
      },
      {
        parameter: 'Dropout',
        Baseline: baseline.dropout,
        Optimized: optimized.dropout
      },
      {
        parameter: 'Frozen Layers',
        Baseline: baseline.frozen_layers,
        Optimized: optimized.frozen_layers
      }
    ]
  }

  const generateAllAlgorithmsComparison = () => {
    // Default values for initial display
    const defaultValues = {
      f1_score: 0,
      accuracy: 0,
      learning_rate: 2e-5,
      batch_size: 16,
      dropout: 0.1,
      frozen_layers: 0,
      training_time: 0
    }
    
    const results = []
    
    // Add PSO - use full data results if available, otherwise optimization results
    const psoData = psoFullDataResults || psoResults?.best_params || psoResults
    results.push({
      algorithm: 'PSO',
      f1_score: psoFullDataResults?.f1_score || psoResults?.best_params?.f1_score || psoResults?.f1_score || defaultValues.f1_score,
      accuracy: psoFullDataResults?.accuracy || psoResults?.best_params?.accuracy || psoResults?.accuracy || defaultValues.accuracy,
      learning_rate: psoData?.learning_rate || psoResults?.best_params?.learning_rate || psoResults?.learning_rate || defaultValues.learning_rate,
      batch_size: psoData?.batch_size || psoResults?.best_params?.batch_size || psoResults?.batch_size || defaultValues.batch_size,
      dropout: psoData?.dropout || psoResults?.best_params?.dropout || psoResults?.dropout || defaultValues.dropout,
      frozen_layers: psoData?.frozen_layers || psoResults?.best_params?.frozen_layers || psoResults?.frozen_layers || defaultValues.frozen_layers,
      training_time: psoResults?.training_time || defaultValues.training_time
    })
    
    // Add GA - use full data results if available, otherwise optimization results
    const gaData = gaFullDataResults || gaResults?.best_params || gaResults
    results.push({
      algorithm: 'GA',
      f1_score: gaFullDataResults?.f1_score || gaResults?.best_params?.f1_score || gaResults?.f1_score || defaultValues.f1_score,
      accuracy: gaFullDataResults?.accuracy || gaResults?.best_params?.accuracy || gaResults?.accuracy || defaultValues.accuracy,
      learning_rate: gaData?.learning_rate || gaResults?.best_params?.learning_rate || gaResults?.learning_rate || defaultValues.learning_rate,
      batch_size: gaData?.batch_size || gaResults?.best_params?.batch_size || gaResults?.batch_size || defaultValues.batch_size,
      dropout: gaData?.dropout || gaResults?.best_params?.dropout || gaResults?.dropout || defaultValues.dropout,
      frozen_layers: gaData?.frozen_layers || gaResults?.best_params?.frozen_layers || gaResults?.frozen_layers || defaultValues.frozen_layers,
      training_time: gaResults?.training_time || defaultValues.training_time
    })
    
    // Add Bayesian - use full data results if available, otherwise optimization results
    const bayesianData = bayesianFullDataResults || bayesianResults?.best_params || bayesianResults
    results.push({
      algorithm: 'Bayesian',
      f1_score: bayesianFullDataResults?.f1_score || bayesianResults?.best_params?.f1_score || bayesianResults?.f1_score || defaultValues.f1_score,
      accuracy: bayesianFullDataResults?.accuracy || bayesianResults?.best_params?.accuracy || bayesianResults?.accuracy || defaultValues.accuracy,
      learning_rate: bayesianData?.learning_rate || bayesianResults?.best_params?.learning_rate || bayesianResults?.learning_rate || defaultValues.learning_rate,
      batch_size: bayesianData?.batch_size || bayesianResults?.best_params?.batch_size || bayesianResults?.batch_size || defaultValues.batch_size,
      dropout: bayesianData?.dropout || bayesianResults?.best_params?.dropout || bayesianResults?.dropout || defaultValues.dropout,
      frozen_layers: bayesianData?.frozen_layers || bayesianResults?.best_params?.frozen_layers || bayesianResults?.frozen_layers || defaultValues.frozen_layers,
      training_time: bayesianResults?.training_time || defaultValues.training_time
    })
    
    return results
  }

  const renderParamMode = (sel: any, param: 'learning_rate'|'batch_size'|'dropout'|'frozen_layers') => {
    if (!sel) return null
    const isOpt = sel.optimize?.[param]
    if (isOpt) {
      return <span className="ml-2 text-xs px-2 py-0.5 rounded bg-green-100 text-green-700">Optimized</span>
    }
    const val = sel.fixed?.[param]
    let display: any = val
    try {
      if (param === 'learning_rate' && typeof val === 'number' && Number.isFinite(val)) {
        display = (val as number).toExponential(2)
      }
    } catch {}
    return <span className="ml-2 text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-700">Fixed: {display ?? 'N/A'}</span>
  }

  const ChartFullscreen = ({ title, children }: { title: string; children: React.ReactNode }) => {
    if (fullscreenChart !== title) return null
    
    return (
      <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg w-full h-full max-w-7xl max-h-[90vh] p-6 overflow-auto">
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
    return <div className="space-y-6" />
  }

  return (
    <div className="space-y-6">
      {/* Training Mode Selection */}
      {mode === 'optimization' && (
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="text-2xl">🎯 Select Optimization Algorithm</CardTitle>
            <CardDescription className="text-base">
              Compare PSO, Genetic Algorithm, and Bayesian Optimization to find the best hyperparameters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button
                onClick={() => setActiveMode('pso')}
                className={`p-6 border-2 rounded-lg transition-all hover:shadow-lg ${
                  activeMode === 'pso'
                    ? 'border-purple-500 bg-purple-50 shadow-md'
                    : 'border-gray-200 hover:border-purple-300'
                }`}
              >
                <Zap className={`h-10 w-10 mb-3 ${activeMode === 'pso' ? 'text-purple-600' : 'text-gray-600'}`} />
                <h3 className="font-bold text-lg mb-2">PSO</h3>
                <p className="text-sm text-muted-foreground mb-2">
                  Particle Swarm Optimization
                </p>
                <div className="text-xs text-gray-500 mt-2 pt-2 border-t">
                  🐝 Swarm Intelligence • ⚡ Efficient
                </div>
              </button>
              
              <button
                onClick={() => setActiveMode('ga')}
                className={`p-6 border-2 rounded-lg transition-all hover:shadow-lg ${
                  activeMode === 'ga'
                    ? 'border-green-500 bg-green-50 shadow-md'
                    : 'border-gray-200 hover:border-green-300'
                }`}
              >
                <TrendingUp className={`h-10 w-10 mb-3 ${activeMode === 'ga' ? 'text-green-600' : 'text-gray-600'}`} />
                <h3 className="font-bold text-lg mb-2">Genetic Algorithm</h3>
                <p className="text-sm text-muted-foreground mb-2">
                  Evolution-based optimization
                </p>
                <div className="text-xs text-gray-500 mt-2 pt-2 border-t">
                  🧬 Natural Selection • 🔄 Diverse
                </div>
              </button>
              
              <button
                onClick={() => setActiveMode('bayesian')}
                className={`p-6 border-2 rounded-lg transition-all hover:shadow-lg ${
                  activeMode === 'bayesian'
                    ? 'border-orange-500 bg-orange-50 shadow-md'
                    : 'border-gray-200 hover:border-orange-300'
                }`}
              >
                <BarChart3 className={`h-10 w-10 mb-3 ${activeMode === 'bayesian' ? 'text-orange-600' : 'text-gray-600'}`} />
                <h3 className="font-bold text-lg mb-2">Bayesian</h3>
                <p className="text-sm text-muted-foreground mb-2">
                  Smart sampling with Optuna
                </p>
                <div className="text-xs text-gray-500 mt-2 pt-2 border-t">
                  📊 Sample-Efficient • 🎯 Intelligent
                </div>
              </button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* PSO Training Parameters */}
      {activeMode === 'pso' && (
        <>
          <Card className="bg-purple-50 border-purple-200">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <Zap className="h-5 w-5 text-purple-600 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-purple-900 mb-1">About PSO</h4>
                  <p className="text-sm text-purple-800">
                    PSO mimics bird flocking behavior. Particles explore the hyperparameter space, 
                    sharing information to converge on optimal solutions. Best for: continuous parameter spaces.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <ParameterOptimizationControls
            algorithm="pso"
            defaultParams={getDefaultParamValues()}
            initialConfig={optimizationConfig ?? undefined}
            isTraining={isTraining}
            onSave={async (params) => {
              try {
                await api.saveOptimizationConfig(params as any)
                await api.saveAlgorithmSettings('pso', psoParams)
                setOptimizationConfig(params as any)
                setToastMessage('Settings saved successfully')
                setTimeout(() => setToastMessage(null), 2000)
              } catch (e) {
                console.error(e)
                setToastMessage('Failed to save settings')
                setTimeout(() => setToastMessage(null), 2500)
              }
            }}
          />

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">🔍 Search Space (Expanded Range)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-2 bg-purple-50 rounded">
                  <p className="font-medium text-purple-900">Learning Rate</p>
                  <p className="text-xs text-purple-700">1e-6 to 1e-4 (100× range)</p>
                </div>
                <div className="p-2 bg-blue-50 rounded">
                  <p className="font-medium text-blue-900">Batch Size</p>
                  <p className="text-xs text-blue-700">4 to 64</p>
                </div>
                <div className="p-2 bg-pink-50 rounded">
                  <p className="font-medium text-pink-900">Dropout</p>
                  <p className="text-xs text-pink-700">0.0 to 0.5</p>
                </div>
                <div className="p-2 bg-cyan-50 rounded">
                  <p className="font-medium text-cyan-900">Frozen Layers</p>
                  <p className="text-xs text-cyan-700">0 to 6</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>PSO Optimization Parameters</CardTitle>
              <CardDescription>Configure Particle Swarm Optimization</CardDescription>
            </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Swarm Size</label>
                <input
                  type="number"
                  value={psoParams.swarmsize}
                  onChange={(e) => setPsoParams({ ...psoParams, swarmsize: parseInt(e.target.value) })}
                  className="w-full p-2 border rounded-md"
                  disabled={isTraining}
                />
                <p className="text-xs text-muted-foreground mt-1">Number of particles in the swarm</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Max Iterations</label>
                <input
                  type="number"
                  value={psoParams.maxiter}
                  onChange={(e) => setPsoParams({ ...psoParams, maxiter: parseInt(e.target.value) })}
                  className="w-full p-2 border rounded-md"
                  disabled={isTraining}
                />
                <p className="text-xs text-muted-foreground mt-1">Maximum number of optimization iterations</p>
              </div>
            </div>
            
            <Button
              onClick={startPSOTraining}
              disabled={isTraining}
              className="w-full mt-4"
            >
              {isTraining ? (
                activeTraining === 'PSO Optimization' ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Optimizing... Iteration {currentIteration}/{psoParams.maxiter}
                  </>
                ) : (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {activeTraining} Running...
                  </>
                )
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" />
                  Optimize
                </>
              )}
            </Button>
            
            {isTraining && (
              <div className="mt-4 space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span>{psoProgress.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all duration-300"
                      style={{ width: `${psoProgress}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground text-center">
                    Evaluating particles ({psoParams.swarmsize} per iteration × {psoParams.maxiter} iterations = {psoParams.swarmsize * psoParams.maxiter} total)
                  </p>
                </div>
                
                {/* Current Particle Metrics */}
                {currentParticleMetrics && (
                  <div className="border-t pt-4">
                    <h4 className="text-sm font-semibold mb-3">Current Particle #{currentParticleMetrics.particle_id}</h4>
                    <div className="grid grid-cols-3 gap-3 mb-3">
                      <div className="p-3 bg-blue-100 rounded-lg text-center">
                        <p className="text-xs text-blue-900">F1 Score</p>
                        <p className="text-lg font-bold text-blue-600">{(currentParticleMetrics.f1_score * 100).toFixed(2)}%</p>
                      </div>
                      <div className="p-3 bg-green-100 rounded-lg text-center">
                        <p className="text-xs text-green-900">Accuracy</p>
                        <p className="text-lg font-bold text-green-600">{(currentParticleMetrics.accuracy * 100).toFixed(2)}%</p>
                      </div>
                      <div className="p-3 bg-purple-100 rounded-lg text-center">
                        <p className="text-xs text-purple-900">Batch Size</p>
                        <p className="text-lg font-bold text-purple-600">{currentParticleMetrics.batch_size}</p>
                      </div>
                    </div>
                    <div className="text-xs space-y-1 text-muted-foreground">
                      <div className="flex justify-between">
                        <span>Learning Rate:</span>
                        <span className="font-mono">{currentParticleMetrics.learning_rate.toExponential(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Dropout:</span>
                        <span className="font-mono">{currentParticleMetrics.dropout.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Frozen Layers:</span>
                        <span className="font-mono">{currentParticleMetrics.frozen_layers}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
        </>
      )}

      {/* GA Training Parameters */}
      {activeMode === 'ga' && (
        <>
          <Card className="bg-green-50 border-green-200">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <TrendingUp className="h-5 w-5 text-green-600 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-green-900 mb-1">About Genetic Algorithm</h4>
                  <p className="text-sm text-green-800">
                    GA mimics natural evolution through selection, crossover, and mutation. 
                    Maintains population diversity while converging to optimal solutions. Best for: complex search spaces.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">🔍 Search Space (Expanded Range)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-2 bg-purple-50 rounded">
                  <p className="font-medium text-purple-900">Learning Rate</p>
                  <p className="text-xs text-purple-700">1e-6 to 1e-4 (100× range)</p>
                </div>
                <div className="p-2 bg-blue-50 rounded">
                  <p className="font-medium text-blue-900">Batch Size</p>
                  <p className="text-xs text-blue-700">4 to 64</p>
                </div>
                <div className="p-2 bg-pink-50 rounded">
                  <p className="font-medium text-pink-900">Dropout</p>
                  <p className="text-xs text-pink-700">0.0 to 0.5</p>
                </div>
                <div className="p-2 bg-cyan-50 rounded">
                  <p className="font-medium text-cyan-900">Frozen Layers</p>
                  <p className="text-xs text-cyan-700">0 to 6</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <ParameterOptimizationControls
            algorithm="ga"
            defaultParams={getDefaultParamValues()}
            initialConfig={optimizationConfig ?? undefined}
            isTraining={isTraining}
            onSave={async (params) => {
              try {
                await api.saveOptimizationConfig(params as any)
                await api.saveAlgorithmSettings('ga', gaParams)
                setOptimizationConfig(params as any)
                setToastMessage('Settings saved successfully')
                setTimeout(() => setToastMessage(null), 2000)
              } catch (e) {
                console.error(e)
                setToastMessage('Failed to save settings')
                setTimeout(() => setToastMessage(null), 2500)
              }
            }}
          />

          <Card>
          <CardHeader>
            <CardTitle>Genetic Algorithm Parameters</CardTitle>
            <CardDescription>Configure evolutionary optimization</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Population Size</label>
                <input
                  type="number"
                  value={gaParams.population_size}
                  onChange={(e) => setGaParams({ ...gaParams, population_size: parseInt(e.target.value) })}
                  className="w-full p-2 border rounded-md"
                  disabled={isTraining}
                />
                <p className="text-xs text-muted-foreground mt-1">Number of individuals in population</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Num Generations</label>
                <input
                  type="number"
                  value={gaParams.num_generations}
                  onChange={(e) => setGaParams({ ...gaParams, num_generations: parseInt(e.target.value) })}
                  className="w-full p-2 border rounded-md"
                  disabled={isTraining}
                />
                <p className="text-xs text-muted-foreground mt-1">Number of evolution generations</p>
              </div>
            </div>
            
            <Button
              onClick={startGATraining}
              disabled={isTraining}
              className="w-full mt-4"
            >
              {isTraining ? (
                activeTraining === 'GA Optimization' ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Evolving... Generation {currentIteration}/{gaParams.num_generations}
                  </>
                ) : (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {activeTraining} Running...
                  </>
                )
              ) : (
                <>
                  <TrendingUp className="mr-2 h-4 w-4" />
                  Optimize
                </>
              )}
            </Button>
            
            {isTraining && (
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{psoProgress.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ width: `${psoProgress}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground text-center">
                  GA is evolving population to find optimal hyperparameters
                </p>
              </div>
            )}
          </CardContent>
        </Card>
        </>
      )}

      {/* Bayesian Optimization Parameters */}
      {activeMode === 'bayesian' && (
        <>
          <Card className="bg-orange-50 border-orange-200">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <BarChart3 className="h-5 w-5 text-orange-600 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-orange-900 mb-1">About Bayesian Optimization</h4>
                  <p className="text-sm text-orange-800">
                    Uses probabilistic models (TPE) to intelligently sample the search space. 
                    Most sample-efficient algorithm - learns from previous trials. Best for: expensive evaluations.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">🔍 Search Space (Expanded Range)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="p-2 bg-purple-50 rounded">
                  <p className="font-medium text-purple-900">Learning Rate</p>
                  <p className="text-xs text-purple-700">1e-6 to 1e-4 (100× range)</p>
                </div>
                <div className="p-2 bg-blue-50 rounded">
                  <p className="font-medium text-blue-900">Batch Size</p>
                  <p className="text-xs text-blue-700">4 to 64</p>
                </div>
                <div className="p-2 bg-pink-50 rounded">
                  <p className="font-medium text-pink-900">Dropout</p>
                  <p className="text-xs text-pink-700">0.0 to 0.5</p>
                </div>
                <div className="p-2 bg-cyan-50 rounded">
                  <p className="font-medium text-cyan-900">Frozen Layers</p>
                  <p className="text-xs text-cyan-700">0 to 6</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <ParameterOptimizationControls
            algorithm="bayesian"
            defaultParams={getDefaultParamValues()}
            initialConfig={optimizationConfig ?? undefined}
            isTraining={isTraining}
            onSave={async (params) => {
              try {
                await api.saveOptimizationConfig(params as any)
                await api.saveAlgorithmSettings('bayesian', bayesianParams)
                setOptimizationConfig(params as any)
                setToastMessage('Settings saved successfully')
                setTimeout(() => setToastMessage(null), 2000)
              } catch (e) {
                console.error(e)
                setToastMessage('Failed to save settings')
                setTimeout(() => setToastMessage(null), 2500)
              }
            }}
          />

          <Card>
          <CardHeader>
            <CardTitle>Bayesian Optimization Parameters</CardTitle>
            <CardDescription>Configure smart sampling with Optuna</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Number of Trials</label>
                <input
                  type="number"
                  value={bayesianParams.n_trials}
                  onChange={(e) => setBayesianParams({ ...bayesianParams, n_trials: parseInt(e.target.value) })}
                  className="w-full p-2 border rounded-md"
                  disabled={isTraining}
                />
                <p className="text-xs text-muted-foreground mt-1">Number of optimization trials (sample-efficient)</p>
              </div>
            </div>
            
            <Button
              onClick={startBayesianTraining}
              disabled={isTraining}
              className="w-full mt-4"
            >
              {isTraining ? (
                activeTraining === 'Bayesian Optimization' ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Optimizing... Trial {currentIteration}/{bayesianParams.n_trials}
                  </>
                ) : (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {activeTraining} Running...
                  </>
                )
              ) : (
                <>
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Optimize
                </>
              )}
            </Button>
            
            {isTraining && (
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{psoProgress.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ width: `${psoProgress}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground text-center">
                  Bayesian optimization is intelligently sampling the hyperparameter space
                </p>
              </div>
            )}
          </CardContent>
        </Card>
        </>
      )}

      {/* Results Cards - Show based on active mode */}
      
      {/* PSO Results */}
      {activeMode === 'pso' && (
        <Card
          key={`pso-${psoResults?.completedAt || psoResults?.best_params?.f1_score || psoResults?.f1_score || 'na'}`}
          className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200"
        >
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
              {psoFullDataResults ? 'PSO Training Results (Full Data)' : 'PSO Optimization Results'}
            </CardTitle>
            <CardDescription>
              {psoFullDataResults ? (
                <>Trained on 100% of data • Completed at {new Date(psoFullDataResults.completedAt).toLocaleString()}</>
              ) : psoResults?.completedAt ? (
                <>Optimization completed at {new Date(psoResults.completedAt).toLocaleString()} • Trained on 20% of data
                {psoResults.training_time && ` • Time: ${(psoResults.training_time / 60).toFixed(1)} min`}</>
              ) : (
                'No PSO optimization run yet'
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Show 20% optimization results - always show if we have results */}
            {psoResults?.best_params && (
              <div className="mb-4 p-3 bg-purple-50 rounded-lg border border-purple-200">
                <p className="text-xs font-semibold text-purple-900 mb-2">Optimization Results (20% of data)</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>F1 Score: <span className="font-semibold">{(psoResults.best_params.f1_score * 100).toFixed(2)}%</span></div>
                  <div>Accuracy: <span className="font-semibold">{(psoResults.best_params.accuracy * 100).toFixed(2)}%</span></div>
                </div>
              </div>
            )}
            
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg text-center">
                <p className="text-sm text-blue-900">F1 Score</p>
                <p className="text-2xl font-bold text-blue-600">
                  {psoFullDataResults ? 
                    `${(psoFullDataResults.f1_score * 100).toFixed(2)}%` :
                    psoResults?.best_params?.f1_score || psoResults?.best_score ? 
                      `${((psoResults.best_params?.f1_score || psoResults.best_score) * 100).toFixed(2)}%` : 
                      '0.00%'
                  }
                </p>
              </div>
              <div className="p-3 bg-green-100 rounded-lg text-center">
                <p className="text-sm text-green-900">Accuracy</p>
                <p className="text-2xl font-bold text-green-600">
                  {psoFullDataResults ? 
                    `${(psoFullDataResults.accuracy * 100).toFixed(2)}%` :
                    psoResults?.best_params?.accuracy ? 
                      `${(psoResults.best_params.accuracy * 100).toFixed(2)}%` : 
                      '0.00%'
                  }
                </p>
              </div>
            </div>
            
            {psoResults?.best_params && (
              <>
                <div className="p-4 bg-white rounded-lg border">
                  <h4 className="font-semibold mb-2">Best Parameters Found</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div><span className="font-medium">Learning Rate:</span> {psoResults.best_params?.learning_rate?.toExponential(2) || 'N/A'}{renderParamMode(psoResults?.parameter_selection,'learning_rate')}</div>
                    <div><span className="font-medium">Batch Size:</span> {psoResults.best_params?.batch_size || 'N/A'}{renderParamMode(psoResults?.parameter_selection,'batch_size')}</div>
                    <div><span className="font-medium">Dropout:</span> {psoResults.best_params?.dropout?.toFixed(3) || 'N/A'}{renderParamMode(psoResults?.parameter_selection,'dropout')}</div>
                    <div><span className="font-medium">Frozen Layers:</span> {psoResults.best_params?.frozen_layers || 'N/A'}{renderParamMode(psoResults?.parameter_selection,'frozen_layers')}</div>
                    <div className="col-span-2 flex items-center gap-2">
                      <span className="font-medium">Epochs:</span>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={psoEpochs}
                        onChange={(e) => setPsoEpochs(parseInt(e.target.value) || 3)}
                        className="w-20 px-2 py-1 border rounded text-sm"
                        disabled={fullDataTraining?.active}
                      />
                    </div>
                  </div>
                </div>
                
                <Button
                  onClick={async () => {
                    try {
                      setFullDataTraining({ active: true, algorithm: 'PSO', progress: 0, message: 'Starting training...' })
                      await api.trainOnFullData('pso', {
                        learning_rate: psoResults.best_params.learning_rate,
                        batch_size: psoResults.best_params.batch_size,
                        dropout: psoResults.best_params.dropout,
                        frozen_layers: psoResults.best_params.frozen_layers,
                        epochs: psoEpochs
                      })
                      // Progress will be tracked by polling
                    } catch (error) {
                      console.error('Error starting full training:', error)
                      setFullDataTraining(null)
                    }
                  }}
                  disabled={fullDataTraining?.active}
                  className="w-full bg-purple-600 hover:bg-purple-700 disabled:opacity-50"
                >
                  {fullDataTraining?.active && fullDataTraining.algorithm === 'PSO' ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Training on Full Data... {fullDataTraining.progress}%
                    </>
                  ) : (
                    <>
                      <Trophy className="mr-2 h-4 w-4" />
                      Train on Full Data with These Parameters
                    </>
                  )}
                </Button>
                
                {/* Full Data Training Progress */}
                {fullDataTraining?.active && fullDataTraining.algorithm === 'PSO' && (
                  <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="font-medium text-purple-900">Training on Full Dataset</span>
                      <span className="font-semibold text-purple-600">{fullDataTraining.progress}%</span>
                    </div>
                    <div className="w-full bg-purple-200 rounded-full h-3">
                      <div
                        className="bg-purple-600 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${fullDataTraining.progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-purple-700 mt-2">{fullDataTraining.message}</p>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}


      {/* GA Results */}
      {activeMode === 'ga' && (
        <Card
          key={`ga-${gaResults?.completedAt || gaResults?.best_params?.f1_score || gaResults?.f1_score || 'na'}`}
          className="bg-gradient-to-br from-green-50 to-green-100 border-green-200"
        >
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
              {gaFullDataResults ? 'GA Training Results (Full Data)' : 'GA Optimization Results'}
            </CardTitle>
            <CardDescription>
              {gaFullDataResults ? (
                <>Trained on 100% of data • Completed at {new Date(gaFullDataResults.completedAt).toLocaleString()}</>
              ) : gaResults?.completedAt ? (
                <>Optimization completed at {new Date(gaResults.completedAt).toLocaleString()} • Trained on 20% of data
                {gaResults.training_time && ` • Time: ${(gaResults.training_time / 60).toFixed(1)} min`}</>
              ) : (
                'No GA optimization run yet'
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Show 20% optimization results - always show if we have results */}
            {gaResults?.best_params && (
              <div className="mb-4 p-3 bg-green-50 rounded-lg border border-green-200">
                <p className="text-xs font-semibold text-green-900 mb-2">Optimization Results (20% of data)</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>F1 Score: <span className="font-semibold">{(gaResults.best_params.f1_score * 100).toFixed(2)}%</span></div>
                  <div>Accuracy: <span className="font-semibold">{(gaResults.best_params.accuracy * 100).toFixed(2)}%</span></div>
                </div>
              </div>
            )}
            
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg text-center">
                <p className="text-sm text-blue-900">F1 Score</p>
                <p className="text-2xl font-bold text-blue-600">
                  {gaFullDataResults ? 
                    `${(gaFullDataResults.f1_score * 100).toFixed(2)}%` :
                    gaResults?.best_params?.f1_score || gaResults?.best_score ? 
                      `${((gaResults.best_params?.f1_score || gaResults.best_score) * 100).toFixed(2)}%` : 
                      '0.00%'
                  }
                </p>
              </div>
              <div className="p-3 bg-green-100 rounded-lg text-center">
                <p className="text-sm text-green-900">Accuracy</p>
                <p className="text-2xl font-bold text-green-600">
                  {gaFullDataResults ? 
                    `${(gaFullDataResults.accuracy * 100).toFixed(2)}%` :
                    gaResults?.best_params?.accuracy ? 
                      `${(gaResults.best_params.accuracy * 100).toFixed(2)}%` : 
                      '0.00%'
                  }
                </p>
              </div>
            </div>
            
            {gaResults?.best_params && (
              <>
                <div className="p-4 bg-white rounded-lg border">
                  <h4 className="font-semibold mb-2">Best Parameters Found</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div><span className="font-medium">Learning Rate:</span> {gaResults.best_params?.learning_rate?.toExponential(2) || 'N/A'}{renderParamMode(gaResults?.parameter_selection,'learning_rate')}</div>
                    <div><span className="font-medium">Batch Size:</span> {gaResults.best_params?.batch_size || 'N/A'}{renderParamMode(gaResults?.parameter_selection,'batch_size')}</div>
                    <div><span className="font-medium">Dropout:</span> {gaResults.best_params?.dropout?.toFixed(3) || 'N/A'}{renderParamMode(gaResults?.parameter_selection,'dropout')}</div>
                    <div><span className="font-medium">Frozen Layers:</span> {gaResults.best_params?.frozen_layers || 'N/A'}{renderParamMode(gaResults?.parameter_selection,'frozen_layers')}</div>
                    <div className="col-span-2 flex items-center gap-2">
                      <span className="font-medium">Epochs:</span>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={gaEpochs}
                        onChange={(e) => setGaEpochs(parseInt(e.target.value) || 3)}
                        className="w-20 px-2 py-1 border rounded text-sm"
                        disabled={fullDataTraining?.active}
                      />
                    </div>
                  </div>
                </div>
                
                <Button
                  onClick={async () => {
                    try {
                      setFullDataTraining({ active: true, algorithm: 'GA', progress: 0, message: 'Starting training...' })
                      await api.trainOnFullData('ga', {
                        learning_rate: gaResults.best_params.learning_rate,
                        batch_size: gaResults.best_params.batch_size,
                        dropout: gaResults.best_params.dropout,
                        frozen_layers: gaResults.best_params.frozen_layers,
                        epochs: gaEpochs
                      })
                      // Progress will be tracked by polling
                    } catch (error) {
                      console.error('Error starting full training:', error)
                      setFullDataTraining(null)
                    }
                  }}
                  disabled={fullDataTraining?.active}
                  className="w-full bg-green-600 hover:bg-green-700 disabled:opacity-50"
                >
                  {fullDataTraining?.active && fullDataTraining.algorithm === 'GA' ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Training on Full Data... {fullDataTraining.progress}%
                    </>
                  ) : (
                    <>
                      <Trophy className="mr-2 h-4 w-4" />
                      Train on Full Data with These Parameters
                    </>
                  )}
                </Button>
                
                {/* Full Data Training Progress */}
                {fullDataTraining?.active && fullDataTraining.algorithm === 'GA' && (
                  <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="font-medium text-green-900">Training on Full Dataset</span>
                      <span className="font-semibold text-green-600">{fullDataTraining.progress}%</span>
                    </div>
                    <div className="w-full bg-green-200 rounded-full h-3">
                      <div
                        className="bg-green-600 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${fullDataTraining.progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-green-700 mt-2">{fullDataTraining.message}</p>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Bayesian Results */}
      {activeMode === 'bayesian' && (
        <Card
          key={`bayes-${bayesianResults?.completedAt || bayesianDisplayF1 || 'na'}`}
          className="bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200"
        >
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
              {bayesianFullDataResults ? 'Bayesian Training Results (Full Data)' : 'Bayesian Optimization Results'}
            </CardTitle>
            <CardDescription>
              {bayesianResults?.completedAt ? (
                <>
                  Optimization completed at {new Date(bayesianResults.completedAt).toLocaleString()} • Trained on 20% of data
                  {typeof bayesianResults.training_time === 'number' && bayesianResults.training_time > 0 ? ` • Time: ${(bayesianResults.training_time / 60).toFixed(1)} min` : ''}
                </>
              ) : (
                'No Bayesian optimization run yet'
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {bayesianHasOptimizationMetrics ? (
              <div className="mb-4 p-3 bg-orange-50 rounded-lg border border-orange-200">
                <p className="text-xs font-semibold text-orange-900 mb-2">Optimization Results (20% of data)</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>F1 Score: <span className="font-semibold">{(bayesianDisplayF1 * 100).toFixed(2)}%</span></div>
                  <div>Accuracy: <span className="font-semibold">{(bayesianDisplayAccuracy * 100).toFixed(2)}%</span></div>
                </div>
              </div>
            ) : (
              <p className="mb-4 text-sm text-gray-600">Bayesian optimization has not produced metrics yet.</p>
            )}

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg text-center">
                <p className="text-sm text-blue-900">F1 Score{bayesianStatLabelSuffix}</p>
                <p className="text-2xl font-bold text-blue-600">{(bayesianMainF1 * 100).toFixed(2)}%</p>
              </div>
              <div className="p-3 bg-green-100 rounded-lg text-center">
                <p className="text-sm text-green-900">Accuracy{bayesianStatLabelSuffix}</p>
                <p className="text-2xl font-bold text-green-600">{(bayesianMainAccuracy * 100).toFixed(2)}%</p>
              </div>
            </div>

            {hasBayesianParams && (
              <>
                <div className="p-4 bg-white rounded-lg border">
                  <h4 className="font-semibold mb-2">Best Parameters Found</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div><span className="font-medium">Learning Rate:</span> {typeof bayesianDisplayParams.learning_rate === 'number' ? bayesianDisplayParams.learning_rate.toExponential(2) : 'N/A'}{renderParamMode(bayesianResults?.parameter_selection,'learning_rate')}</div>
                    <div><span className="font-medium">Batch Size:</span> {bayesianDisplayParams.batch_size ?? 'N/A'}{renderParamMode(bayesianResults?.parameter_selection,'batch_size')}</div>
                    <div><span className="font-medium">Dropout:</span> {typeof bayesianDisplayParams.dropout === 'number' ? bayesianDisplayParams.dropout.toFixed(3) : 'N/A'}{renderParamMode(bayesianResults?.parameter_selection,'dropout')}</div>
                    <div><span className="font-medium">Frozen Layers:</span> {bayesianDisplayParams.frozen_layers ?? 'N/A'}{renderParamMode(bayesianResults?.parameter_selection,'frozen_layers')}</div>
                    <div className="col-span-2 flex items-center gap-2">
                      <span className="font-medium">Epochs:</span>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={bayesianEpochs}
                        onChange={(e) => setBayesianEpochs(parseInt(e.target.value) || 3)}
                        className="w-20 px-2 py-1 border rounded text-sm"
                        disabled={fullDataTraining?.active}
                      />
                    </div>
                  </div>
                </div>

                <Button
                  onClick={async () => {
                    if (!bayesianDisplayParams) return
                    try {
                      setFullDataTraining({ active: true, algorithm: 'Bayesian', progress: 0, message: 'Starting training...' })
                      await api.trainOnFullData('bayesian', {
                        learning_rate: Number(bayesianDisplayParams.learning_rate ?? bayesianResults?.best_params?.learning_rate),
                        batch_size: Number(bayesianDisplayParams.batch_size ?? bayesianResults?.best_params?.batch_size),
                        dropout: Number(bayesianDisplayParams.dropout ?? bayesianResults?.best_params?.dropout),
                        frozen_layers: Number(bayesianDisplayParams.frozen_layers ?? bayesianResults?.best_params?.frozen_layers),
                        epochs: bayesianEpochs
                      })
                    } catch (error) {
                      console.error('Error starting full training:', error)
                      setFullDataTraining(null)
                    }
                  }}
                  disabled={fullDataTraining?.active}
                  className="w-full bg-orange-600 hover:bg-orange-700 disabled:opacity-50"
                >
                  {fullDataTraining?.active && fullDataTraining.algorithm === 'Bayesian' ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Training on Full Data... {fullDataTraining.progress}%
                    </>
                  ) : (
                    <>
                      <Trophy className="mr-2 h-4 w-4" />
                      Train on Full Data with These Parameters
                    </>
                  )}
                </Button>

                {fullDataTraining?.active && fullDataTraining.algorithm === 'Bayesian' && (
                  <div className="mt-4 p-4 bg-orange-50 rounded-lg border border-orange-200">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="font-medium text-orange-900">Training on Full Dataset</span>
                      <span className="font-semibold text-orange-600">{fullDataTraining.progress}%</span>
                    </div>
                    <div className="w-full bg-orange-200 rounded-full h-3">
                      <div
                        className="bg-orange-600 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${fullDataTraining.progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-orange-700 mt-2">{fullDataTraining.message}</p>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}
      {/* PSO Comprehensive Visualization - Only show in PSO tab */}
      {activeMode === 'pso' && psoAnimation.length > 0 && (
        <PSOVisualization psoHistory={psoHistory} psoAnimation={psoAnimation} />
      )}

      {/* GA Results Display */}
      {gaAnimation.length > 0 && gaHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-6 w-6 text-green-600" />
              Genetic Algorithm Optimization Complete
            </CardTitle>
            <CardDescription>
              Evaluated {gaAnimation.length} individuals • Population: {gaParams.population_size} • Generations: {gaParams.num_generations}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {gaHistory.length > 0 && gaHistory[gaHistory.length - 1].best_params && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-lg">
                  <p className="text-sm font-medium text-blue-900 mb-1">Best F1 Score</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {(gaHistory[gaHistory.length - 1].best_params.f1_score * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-lg">
                  <p className="text-sm font-medium text-green-900 mb-1">Accuracy</p>
                  <p className="text-3xl font-bold text-green-600">
                    {(gaHistory[gaHistory.length - 1].best_params.accuracy * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-lg">
                  <p className="text-sm font-medium text-purple-900 mb-1">Learning Rate</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {gaHistory[gaHistory.length - 1].best_params.learning_rate?.toFixed(6)}
                  </p>
                </div>
                <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 border border-orange-200 rounded-lg">
                  <p className="text-sm font-medium text-orange-900 mb-1">Batch Size</p>
                  <p className="text-3xl font-bold text-orange-600">
                    {gaHistory[gaHistory.length - 1].best_params.batch_size}
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* All Algorithms Comparison */}
      {generateAllAlgorithmsComparison().length > 0 && (
        <Card className="border-2 border-primary">
          <CardHeader>
            <CardTitle className="text-2xl">🏆 All Algorithms Comparison</CardTitle>
            <CardDescription>
              Performance metrics and hyperparameters across all training methods
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-8">
              {/* Plot 1: Performance Metrics (F1, Accuracy, Loss, Time) */}
              <div>
                <h3 className="text-lg font-semibold mb-4">📊 Performance Metrics</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Accuracy & F1 Score */}
                  <div>
                    <p className="text-sm font-medium mb-2 text-center">Accuracy & F1 Score</p>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={generateAllAlgorithmsComparison()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip formatter={(value: any) => `${(value * 100).toFixed(2)}%`} />
                        <Legend />
                        <Bar dataKey="f1_score" name="F1 Score" fill="#3b82f6" />
                        <Bar dataKey="accuracy" name="Accuracy" fill="#10b981" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  
                  {/* Optimization Time */}
                  <div>
                    <p className="text-sm font-medium mb-2 text-center">Optimization Time (minutes)</p>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={generateAllAlgorithmsComparison().map(item => ({
                        ...item,
                        time_minutes: (item.training_time || 0) / 60
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis />
                        <Tooltip formatter={(value: any) => `${value.toFixed(1)} min`} />
                        <Legend />
                        <Bar dataKey="time_minutes" name="Time (min)" fill="#f59e0b" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                {/* Winner Badge */}
                {(() => {
                  const results = generateAllAlgorithmsComparison()
                  if (results.length === 0) return null
                  const winner = results.reduce((best, current) => 
                    current.f1_score > best.f1_score ? current : best
                  )
                  return (
                    <div className="mt-4 p-4 bg-gradient-to-r from-yellow-50 to-yellow-100 border-2 border-yellow-400 rounded-lg">
                      <div className="flex items-center gap-3">
                        <Trophy className="h-8 w-8 text-yellow-600" />
                        <div>
                          <p className="text-lg font-bold text-yellow-900">Best Performing: {winner.algorithm}</p>
                          <p className="text-sm text-yellow-800">
                            F1: {(winner.f1_score * 100).toFixed(2)}% • 
                            Accuracy: {(winner.accuracy * 100).toFixed(2)}% • 
                            Time: {((winner.training_time || 0) / 60).toFixed(1)} min
                          </p>
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </div>

              {/* Plot 2: Hyperparameters */}
              <div>
                <h3 className="text-lg font-semibold mb-4">⚙️ Hyperparameters Chosen</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Learning Rate */}
                  <div>
                    <p className="text-sm font-medium mb-2">Learning Rate</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={generateAllAlgorithmsComparison()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="learning_rate" fill="#8b5cf6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Batch Size */}
                  <div>
                    <p className="text-sm font-medium mb-2">Batch Size</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={generateAllAlgorithmsComparison()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="batch_size" fill="#f59e0b" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Dropout */}
                  <div>
                    <p className="text-sm font-medium mb-2">Dropout</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={generateAllAlgorithmsComparison()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="dropout" fill="#ec4899" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Frozen Layers */}
                  <div>
                    <p className="text-sm font-medium mb-2">Frozen Layers</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={generateAllAlgorithmsComparison()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="frozen_layers" fill="#06b6d4" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Detailed Table */}
              <div>
                <h3 className="text-lg font-semibold mb-4">Detailed Comparison Table</h3>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-gray-100">
                        <th className="border p-2 text-left font-semibold">Algorithm</th>
                        <th className="border p-2 text-center font-semibold">F1 Score</th>
                        <th className="border p-2 text-center font-semibold">Accuracy</th>
                        <th className="border p-2 text-center font-semibold">Learning Rate</th>
                        <th className="border p-2 text-center font-semibold">Batch Size</th>
                        <th className="border p-2 text-center font-semibold">Dropout</th>
                        <th className="border p-2 text-center font-semibold">Frozen Layers</th>
                      </tr>
                    </thead>
                    <tbody>
                      {generateAllAlgorithmsComparison().map((result, idx) => (
                        <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          <td className="border p-2 font-semibold">{result.algorithm}</td>
                          <td className="border p-2 text-center">{(result.f1_score * 100).toFixed(2)}%</td>
                          <td className="border p-2 text-center">{(result.accuracy * 100).toFixed(2)}%</td>
                          <td className="border p-2 text-center">{result.learning_rate?.toFixed(6)}</td>
                          <td className="border p-2 text-center">{result.batch_size}</td>
                          <td className="border p-2 text-center">{result.dropout?.toFixed(3)}</td>
                          <td className="border p-2 text-center">{result.frozen_layers}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Winner Badge */}
              {(() => {
                const results = generateAllAlgorithmsComparison()
                if (results.length === 0) return null
                const winner = results.reduce((best, current) => 
                  current.f1_score > best.f1_score ? current : best
                )
                return (
                  <div className="mt-4 p-4 bg-gradient-to-r from-yellow-50 to-yellow-100 border-2 border-yellow-400 rounded-lg">
                    <div className="flex items-center gap-3">
                      <span className="text-4xl">🏆</span>
                      <div>
                        <h4 className="text-lg font-bold text-yellow-900">Best Performing Algorithm</h4>
                        <p className="text-yellow-800">
                          <span className="font-bold text-xl">{winner.algorithm}</span> achieved the highest F1 Score of{' '}
                          <span className="font-bold text-xl">{(winner.f1_score * 100).toFixed(2)}%</span>
                        </p>
                      </div>
                    </div>
                  </div>
                )
              })()}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Training History Log */}
      {trainingLogs.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <div>
                <CardTitle>Training History</CardTitle>
                <CardDescription>
                  {lastTrainingTime && `Last training: ${new Date(lastTrainingTime).toLocaleString()}`}
                </CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={clearTrainingLogs}
              >
                Clear History
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {trainingLogs.map((log) => (
                <div
                  key={log.id}
                  className={`p-4 rounded-lg border-2 ${
                    log.status === 'completed'
                      ? 'border-green-200 bg-green-50'
                      : 'border-red-200 bg-red-50'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <h4 className="font-semibold text-lg flex items-center gap-2">
                        {log.mode}
                        {log.status === 'completed' ? (
                          <span className="text-green-600 text-sm">✓ Completed</span>
                        ) : (
                          <span className="text-red-600 text-sm">✗ Failed</span>
                        )}
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        {new Date(log.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-3">
                    <p className="text-sm font-medium mb-2">Parameters:</p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                      {Object.entries(log.parameters).map(([key, value]) => (
                        <div key={key} className="bg-white/50 p-2 rounded">
                          <span className="font-medium">{key.replace(/_/g, ' ')}:</span>{' '}
                          <span>{typeof value === 'number' ? value.toFixed(6) : String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {log.error && (
                    <div className="mt-3 p-2 bg-red-100 rounded text-sm text-red-800">
                      <strong>Error:</strong> {log.error}
                    </div>
                  )}
                  
                  {log.result && (
                    <div className="mt-3 p-2 bg-white/50 rounded text-sm">
                      <strong>Result:</strong> Training process started (PID: {log.result.pid})
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Fullscreen Charts */}
      <ChartFullscreen title="Hyperparameter Comparison">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={generateComparisonData()}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="parameter" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="Baseline" fill="#94a3b8" />
            <Bar dataKey="Optimized" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </ChartFullscreen>

      <ChartFullscreen title="PSO Convergence History">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={psoHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
            <YAxis label={{ value: 'F1 Score', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="best_score" stroke="#3b82f6" strokeWidth={3} name="Best F1 Score" />
            <Line type="monotone" dataKey="avg_score" stroke="#94a3b8" strokeWidth={3} name="Average F1 Score" />
          </LineChart>
        </ResponsiveContainer>
      </ChartFullscreen>

      <ChartFullscreen title="PSO Particle Movement">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="learning_rate" name="Learning Rate" />
            <YAxis dataKey="dropout" name="Dropout" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            <Scatter name="Particles" data={psoAnimation} fill="#3b82f6">
              {psoAnimation.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.is_best ? '#ef4444' : '#3b82f6'} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </ChartFullscreen>
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50">
          <div className="px-4 py-3 rounded-md shadow-md text-white bg-emerald-600">
            {toastMessage}
          </div>
        </div>
      )}
    </div>
  )
}
