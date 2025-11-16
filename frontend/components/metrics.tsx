"use client"

// Coleta leve de LCP, INP (fallback FID) e TTFB e envia para console.
// Pode ser facilmente adaptado para enviar a um endpoint de métricas.
import { useEffect } from "react"

export default function Metrics() {
  useEffect(() => {
    try {
      // TTFB
      const nav = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined
      if (nav) {
        const ttfb = nav.responseStart
        console.info('[metrics] TTFB(ms)=', Math.round(ttfb))
      }

      // LCP
      if ('PerformanceObserver' in window) {
        try {
          const po = new PerformanceObserver((list) => {
            const entries = list.getEntries()
            const last = entries[entries.length - 1] as any
            if (last) {
              const lcp = last.startTime
              console.info('[metrics] LCP(ms)=', Math.round(lcp))
            }
          })
          po.observe({ type: 'largest-contentful-paint', buffered: true as any })
        } catch {}

        // FID (fallback) ou INP quando suportado: name = 'first-input' (antigo)
        try {
          const po2 = new PerformanceObserver((list) => {
            const entry = list.getEntries()[0] as any
            if (entry) {
              const fid = entry.processingStart - entry.startTime
              console.info('[metrics] FID(ms)=', Math.round(fid))
            }
          })
          po2.observe({ type: 'first-input', buffered: true as any })
        } catch {}
      }
    } catch {}
  }, [])
  return null
}
