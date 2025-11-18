import Navbar from '@/app/components/Navbar';
import ModelComparison from '@/app/components/ModelComparison';

export default function PerformancePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Navbar />
      
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-center mb-12">Model Performance</h1>
          
          <div className="bg-white rounded-xl shadow-sm p-6 mb-8 border border-gray-100">
            <h2 className="text-2xl font-semibold mb-6">Model Comparison</h2>
            <ModelComparison />
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
              <h2 className="text-xl font-semibold mb-4">Accuracy Metrics</h2>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Precision</span>
                    <span className="text-sm text-gray-500">92.5%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: '92.5%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Recall</span>
                    <span className="text-sm text-gray-500">89.3%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div className="bg-green-600 h-2.5 rounded-full" style={{ width: '89.3%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">F1 Score</span>
                    <span className="text-sm text-gray-500">90.8%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div className="bg-purple-600 h-2.5 rounded-full" style={{ width: '90.8%' }}></div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
              <h2 className="text-xl font-semibold mb-4">Model Information</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Model Type</span>
                  <span className="font-medium">DistilBERT</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Parameters</span>
                  <span className="font-medium">66M</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Training Data</span>
                  <span className="font-medium">1M+ samples</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Optimization</span>
                  <span className="font-medium">Metaheuristic</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Inference Time</span>
                  <span className="font-medium">~50ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
