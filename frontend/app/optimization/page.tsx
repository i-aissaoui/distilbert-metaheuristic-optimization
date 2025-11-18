'use client';

import Navbar from '@/app/components/Navbar';
import dynamic from 'next/dynamic';

const TrainingClient = dynamic(() => import('@/app/components/Training'), { ssr: false });

export default function OptimizationPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Navbar />
      
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-center mb-8">Model Optimization</h1>
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <TrainingClient mode="optimization" />
          </div>
        </div>
      </main>
    </div>
  );
}
