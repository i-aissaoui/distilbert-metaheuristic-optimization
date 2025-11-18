'use client'

import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { type Locale, isLocale, translate } from './messages'

type I18nContextType = {
  locale: Locale
  setLocale: (l: Locale) => void
  t: (key: string) => string
}

const I18nContext = createContext<I18nContextType | undefined>(undefined)

export function I18nProvider({ children, initialLocale }: { children: React.ReactNode; initialLocale?: Locale }) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale ?? 'en')

  useEffect(() => {
    // No persistence: locale is in-memory only
  }, [])

  const setLocale = (l: Locale) => {
    setLocaleState(l)
    // No persistence to localStorage or cookies
  }

  const t = useMemo(() => {
    return (key: string) => translate(locale, key)
  }, [locale])

  const value = useMemo(() => ({ locale, setLocale, t }), [locale, t])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n must be used within I18nProvider')
  return ctx
}
