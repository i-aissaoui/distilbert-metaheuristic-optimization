import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface PredictionRequest {
  text: string
  use_optimized?: boolean
  model_type?: 'pso' | 'ga' | 'bayesian'
}

export interface PredictionResponse {
  text: string
  predicted_label: string
  confidence: number
  all_probabilities: Record<string, number>
  model_used: string
}

export interface ModelInfo {
  model_name: string
  num_labels: number
  labels: string[]
  baseline_metrics?: {
    accuracy: number
    f1_weighted: number
    f1_macro: number
    precision: number
    recall: number
  }
  optimized_metrics?: {
    accuracy: number
    f1_weighted: number
    f1_macro: number
    precision: number
    recall: number
  }
  improvement?: {
    accuracy: number
    f1_weighted: number
    f1_macro: number
  }
  improvement_percentage?: {
    accuracy: number
    f1_weighted: number
    f1_macro: number
  }
}

export interface HealthResponse {
  status: string
  models_loaded: {
    baseline: boolean
    optimized: boolean
  }
}

export interface TrainingParams {
  learning_rate: number
  batch_size: number
  dropout: number
  frozen_layers: number
  epochs: number
}

export interface PSOParams {
  swarmsize: number
  maxiter: number
}

export interface GAParams {
  population_size: number
  num_generations: number
}

export interface BayesianParams {
  n_trials: number
}

export interface OptimizationConfig {
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
}

export interface TrainingStatus {
  status: string
  message?: string
  progress?: number
  current_step?: number
  total_steps?: number
  current_epoch?: number
  total_epochs?: number
  loss?: number
  accuracy?: number
  f1_score?: number
  metrics?: any
  iteration?: number
  generation?: number
  trial?: number
  best_score?: number
  best_params?: any
  total_particles?: number
  total_iterations?: number
  total_individuals?: number
  total_trials?: number
  current_particle_metrics?: {
    particle_id: number
    f1_score: number
    accuracy: number
    learning_rate: number
    batch_size: number
    dropout: number
    frozen_layers: number
  }
  best_so_far?: {
    f1_score: number
    params: any
  }
}

export const api = {
  async predict(data: PredictionRequest): Promise<PredictionResponse> {
    const response = await axios.post(`${API_URL}/predict`, data)
    return response.data
  },

  async getModelInfo(): Promise<ModelInfo> {
    const response = await axios.get(`${API_URL}/model-info`)
    return response.data
  },

  async getLabels(): Promise<{ labels: string[]; descriptions: Record<string, string> }> {
    const response = await axios.get(`${API_URL}/labels`)
    return response.data
  },

  async healthCheck(): Promise<HealthResponse> {
    const response = await axios.get(`${API_URL}/health`)
    return response.data
  },

  async startTraining(params: TrainingParams): Promise<any> {
    const response = await axios.post(`${API_URL}/start-training`, null, { 
      params,
      timeout: 30000 // 30 second timeout
    })
    return response.data
  },

  async startPSO(params: PSOParams): Promise<any> {
    const response = await axios.post(`${API_URL}/start-pso`, null, { params })
    return response.data
  },

  async startGA(params: GAParams): Promise<any> {
    const response = await axios.post(`${API_URL}/start-ga`, null, { params })
    return response.data
  },

  async startBayesian(params: BayesianParams): Promise<any> {
    const response = await axios.post(`${API_URL}/start-bayesian`, null, { params })
    return response.data
  },

  async getTrainingStatus(): Promise<TrainingStatus> {
    const response = await axios.get(`${API_URL}/training-status`)
    return response.data
  },

  async saveTrainingResult(algorithmType: string, result: any): Promise<void> {
    // Send result directly, not nested
    await axios.post(`${API_URL}/save-result`, result)
  },

  async loadTrainingResult(algorithmType: string): Promise<any> {
    const response = await axios.get(`${API_URL}/load-result/${algorithmType}`)
    return response.data
  },

  async clearTrainingHistory(): Promise<void> {
    await axios.post(`${API_URL}/clear-history`)
  },

  async getTrainingHistory(): Promise<any> {
    const response = await axios.get(`${API_URL}/training-history`)
    return response.data
  },

  async getPSOHistory(): Promise<any> {
    const response = await axios.get(`${API_URL}/pso-history`)
    return response.data
  },

  async getPSOAnimation(): Promise<any> {
    const response = await axios.get(`${API_URL}/pso-animation`)
    return response.data
  },

  async getGAHistory(): Promise<any> {
    const response = await axios.get(`${API_URL}/ga-history`)
    return response.data
  },

  async getGAAnimation(): Promise<any> {
    const response = await axios.get(`${API_URL}/ga-animation`)
    return response.data
  },

  async getBayesianHistory(): Promise<any> {
    const response = await axios.get(`${API_URL}/bayesian-history`)
    return response.data
  },

  async getBayesianAnimation(): Promise<any> {
    const response = await axios.get(`${API_URL}/bayesian-animation`)
    return response.data
  },

  async getHyperparameters(): Promise<any> {
    const response = await axios.get(`${API_URL}/hyperparameters`)
    return response.data
  },

  async getOptimizationConfig(): Promise<OptimizationConfig> {
    const response = await axios.get(`${API_URL}/optimization-config`)
    return response.data
  },

  async saveOptimizationConfig(config: OptimizationConfig): Promise<any> {
    const response = await axios.post(`${API_URL}/optimization-config`, config)
    return response.data
  },

  async getAlgorithmSettings(): Promise<any> {
    const response = await axios.get(`${API_URL}/algorithm-settings`)
    return response.data
  },

  async saveAlgorithmSettings(algorithm: 'pso' | 'ga' | 'bayesian', settings: any): Promise<any> {
    const response = await axios.post(`${API_URL}/algorithm-settings`, { algorithm, settings })
    return response.data
  },

  async trainOnFullData(algorithm: string, params: TrainingParams): Promise<any> {
    const response = await axios.post(`${API_URL}/train-full-data`, null, {
      params: {
        algorithm,
        ...params
      }
    })
    return response.data
  },
}
