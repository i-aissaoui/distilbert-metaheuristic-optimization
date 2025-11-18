"use client";
import Navbar from '@/app/components/Navbar';
import PredictionForm from '@/app/components/PredictionForm';
import PredictionResult from '@/app/components/PredictionResult';
import { useState } from 'react';
import { PredictionResponse } from '@/lib/api';

export default function AnalyzePage() {
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Navbar />
      
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-center mb-12">Text Analysis</h1>
          
          <div className="bg-white rounded-xl shadow-sm p-6 mb-8 border border-gray-100">
            <h2 className="text-xl font-semibold mb-4">Enter Text to Analyze</h2>
            <PredictionForm 
              onPrediction={setPredictionResult}
            />
            
            {predictionResult && (
              <div className="mt-8">
                <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
                <PredictionResult result={predictionResult} />
              </div>
            )}
          </div>
          
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <h2 className="text-xl font-semibold mb-4">About the Analysis</h2>
            <p className="text-gray-600 mb-4">
              Our hate speech detection system uses a fine-tuned DistilBERT model optimized with metaheuristic algorithms 
              to identify potentially harmful content with high accuracy.
            </p>
            <p className="text-gray-600">
              The model has been trained on diverse datasets to recognize various forms of hate speech, including 
              explicit and implicit forms of discrimination, harassment, and harmful content.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
