import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

export function getLabelColor(label: string): string {
  const colors: Record<string, string> = {
    'hate': 'text-red-600 bg-red-50 border-red-200',
    'not-hate': 'text-green-600 bg-green-50 border-green-200',
    '1': 'text-red-600 bg-red-50 border-red-200',
    '0': 'text-green-600 bg-green-50 border-green-200',
  }
  return colors[label] || 'text-gray-600 bg-gray-50 border-gray-200'
}

export function getLabelIcon(label: string): string {
  const icons: Record<string, string> = {
    'hate': '🚫',
    'not-hate': '✅',
    '1': '🚫',
    '0': '✅',
  }
  return icons[label] || '❓'
}
