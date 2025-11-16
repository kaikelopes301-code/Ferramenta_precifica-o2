"use client"

import { useRef, useState, useEffect } from "react"

interface HorizontalScrollProps {
  children: React.ReactNode
  itemMinWidth?: number // px
}

export function HorizontalScroll({ children, itemMinWidth = 280 }: HorizontalScrollProps) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [drag, setDrag] = useState<{active:boolean; x:number; scroll:number}>({active:false, x:0, scroll:0})
  const rafRef = useRef<number | null>(null)
  const momentumRef = useRef<number | null>(null)
  const lastScrollRef = useRef(0)
  const lastTimeRef = useRef(0)
  const velocityRef = useRef(0)

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = ref.current
    if (!el) return
    // Evita iniciar drag quando clicar em elementos interativos (botões, links, inputs)
    const targetEl = e.target as HTMLElement
    if (targetEl && targetEl.closest('button, a, input, textarea, select, [data-interactive]')) {
      return
    }
    if (e.button !== 0) return
    el.setPointerCapture(e.pointerId)
    setDrag({ active: true, x: e.clientX, scroll: el.scrollLeft })
    lastScrollRef.current = el.scrollLeft
    lastTimeRef.current = performance.now()
    velocityRef.current = 0
    if (momentumRef.current) {
      cancelAnimationFrame(momentumRef.current)
      momentumRef.current = null
    }
  }
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drag.active) return
    const el = ref.current
    if (!el) return
    const now = performance.now()
    const dx = e.clientX - drag.x
    const nextScroll = drag.scroll - dx
    const prevScroll = lastScrollRef.current
    el.scrollLeft = nextScroll
    const dt = Math.max(1, now - lastTimeRef.current)
    const v = (el.scrollLeft - prevScroll) / dt // px por ms
    velocityRef.current = v
    lastScrollRef.current = el.scrollLeft
    lastTimeRef.current = now
  }
  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = ref.current
    if (!el) return
    try { el.releasePointerCapture(e.pointerId) } catch {}
    setDrag(prev => ({ ...prev, active: false }))
    // Momentum: continua rolando baseado na velocidade atual
    const startV = velocityRef.current
    const friction = 0.94 // 0-1 (menor = mais atrito)
    const minV = 0.02 // px/ms
    let v = startV
    let prevTs = performance.now()
    const step = () => {
      const el2 = ref.current
      if (!el2) return
      const ts = performance.now()
      const dt = Math.min(32, ts - prevTs) // limita para estabilidade
      prevTs = ts
      v *= Math.pow(friction, dt / 16)
      if (Math.abs(v) < minV) { momentumRef.current = null; return }
      const maxScroll = el2.scrollWidth - el2.clientWidth
      el2.scrollLeft = Math.max(0, Math.min(maxScroll, el2.scrollLeft + v * dt))
      scheduleUpdate()
      momentumRef.current = requestAnimationFrame(step)
    }
    if (Math.abs(startV) > minV) {
      momentumRef.current = requestAnimationFrame(step)
    }
  }

  const scheduleUpdate = () => {
    if (rafRef.current) return
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null
      const el = ref.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const center = rect.left + rect.width / 2
      const items = el.querySelectorAll<HTMLDivElement>('[data-item]')
      items.forEach((it) => {
        const r = it.getBoundingClientRect()
        const c = r.left + r.width / 2
        const dx = (c - center) / rect.width // -0.5..0.5 aprox
        const dist = Math.min(1, Math.abs(dx) * 2) // 0 centro, 1 borda
        const scale = 0.985 + (1 - dist) * 0.015 // 0.985..1.0 (variação menor)
        const lift = (1 - dist) * 4 // 0..4px (mais sutil)
        it.style.willChange = 'transform'
        it.style.transform = `translateY(${-lift}px) scale(${scale})`
      })
    })
  }

  const onScroll = () => scheduleUpdate()

  // Atualiza na montagem e em resize
  useEffect(() => {
    scheduleUpdate()
    const onResize = () => scheduleUpdate()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  return (
    <div className="relative">
      <div
        ref={ref}
        className="hide-scrollbar overflow-x-auto overflow-y-visible snap-x snap-mandatory scroll-p-4 drag-no-select cursor-grab active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onScroll}
        onScroll={onScroll}
      >
        <div className="flex gap-6 px-1">
          {Array.isArray(children)
            ? children.map((child, i) => (
                <div
                  key={i}
                  className="snap-start shrink-0 transition-transform duration-300 ease-out"
                  style={{ minWidth: `${itemMinWidth}px` }}
                >
                  <div className="animate-carousel-in" style={{ animationDelay: `${i * 60}ms` }} data-item>{child}</div>
                </div>
              ))
            : (
              <div className="snap-start shrink-0 transition-transform duration-300 ease-out" style={{ minWidth: `${itemMinWidth}px` }} data-item>{children}</div>
            )}
        </div>
      </div>
      {/* Gradientes laterais removidos para evitar sombra escura; mantenha apenas caso queira pistas de overflow. */}
    </div>
  )
}
