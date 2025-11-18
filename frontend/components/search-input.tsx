"use client"

import type React from "react"

import { useEffect, useRef, useState } from "react"
import { Search, Loader2, Send, Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip"

interface SearchOptions {
  topK: number
  useTfidf: boolean
}

interface SearchInputProps {
  onSearch: (description: string, options: SearchOptions) => void
  isLoading: boolean
}

export function SearchInput({ onSearch, isLoading }: SearchInputProps) {
  const [value, setValue] = useState("")
  const [topK, setTopK] = useState(5)
  // TF-IDF descontinuado na UI; mantemos flag para compatibilidade, fixo em false
  const [useTfidf] = useState(false)
  const [showOptions, setShowOptions] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const [typedPlaceholder, setTypedPlaceholder] = useState("")
  const typingIndexRef = useRef(0)
  const typingTimerRef = useRef<number | null>(null)
  const placeholderFull =
    "Descreva os equipamentos que você precisa... (ex.: vassoura profissional, mop industrial, aspirador de pó)"

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (value.trim() && !isLoading) {
      // Envia o texto completo - o page.tsx vai fazer o split
      onSearch(value.trim(), { topK, useTfidf })
    }
  }

  // Auto-ajuste da altura do textarea: cresce até 3 linhas, depois rola
  const autoResize = () => {
    const el = textareaRef.current
    if (!el) return
    // Reset para medir corretamente
    el.style.height = 'auto'
    // Calcula altura máxima baseada em 3 linhas + paddings
    const cs = window.getComputedStyle(el)
    const lineH = parseFloat(cs.lineHeight || '0') || 24
    const padTop = parseFloat(cs.paddingTop || '0') || 0
    const padBottom = parseFloat(cs.paddingBottom || '0') || 0
    const maxH = Math.ceil(lineH * 3 + padTop + padBottom)
    const next = Math.min(el.scrollHeight, maxH)
    el.style.height = `${next}px`
    el.style.overflowY = el.scrollHeight > maxH ? 'auto' : 'hidden'
  }

  useEffect(() => {
    autoResize()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  // Animação de digitação do placeholder ao carregar
  useEffect(() => {
    // Respeita preferência do usuário
    const reduce = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce) {
      setTypedPlaceholder(placeholderFull)
      return
    }
    // Evita animar se o usuário já começou a digitar
    if (value.trim().length > 0) {
      setTypedPlaceholder(placeholderFull)
      return
    }

    // Função de digitação incremental
    const typeNext = () => {
      // Se usuário começou a digitar, interrompe
      if (value.trim().length > 0) {
        setTypedPlaceholder(placeholderFull)
        typingTimerRef.current && window.clearTimeout(typingTimerRef.current)
        typingTimerRef.current = null
        return
      }
      const i = typingIndexRef.current
      if (i >= placeholderFull.length) {
        typingTimerRef.current && window.clearTimeout(typingTimerRef.current)
        typingTimerRef.current = null
        return
      }
      setTypedPlaceholder(placeholderFull.slice(0, i + 1))
      typingIndexRef.current = i + 1
      // Leve variação para sensação mais natural
      const delay = 18 + Math.floor(Math.random() * 30) // 18–48ms
      typingTimerRef.current = window.setTimeout(typeNext, delay)
    }

    // Inicializa
    typingIndexRef.current = 0
    setTypedPlaceholder("")
    typingTimerRef.current = window.setTimeout(typeNext, 200) // pequeno atraso inicial

    // Cleanup
    return () => {
      if (typingTimerRef.current) {
        window.clearTimeout(typingTimerRef.current)
        typingTimerRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
  <div className="app-container max-w-5xl">
      <form onSubmit={handleSubmit} className="group">
        <div className="relative rounded-full bg-card/95 backdrop-blur-md border-2 border-border shadow-xl ring-0 transition-all duration-300 hover:shadow-2xl focus-within:shadow-2xl focus-within:border-primary/60 focus-within:ring-4 focus-within:ring-primary/20">
          <div className="pointer-events-none absolute left-6 top-[calc(50%+1px)] -translate-y-1/2 flex items-center">
            <Search className="h-5 w-5 text-muted-foreground transition-colors group-focus-within:text-primary" />
          </div>
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onInput={autoResize}
            onKeyDown={(e) => {
              // Ctrl+Enter ou Cmd+Enter para enviar
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault()
                if (value.trim() && !isLoading) {
                  onSearch(value.trim(), { topK, useTfidf })
                }
              }
              // Enter simples = quebra de linha (comportamento padrão do textarea)
            }}
            onFocus={() => {
              // Se focou e ainda está animando, finalize imediatamente
              if (typingTimerRef.current) {
                window.clearTimeout(typingTimerRef.current)
                typingTimerRef.current = null
                setTypedPlaceholder(placeholderFull)
              }
            }}
            placeholder={typedPlaceholder}
            aria-label="Campo de busca de equipamentos. Pressione Ctrl+Enter para buscar"
            title="Enter = quebra de linha | Ctrl+Enter = buscar"
            className="w-full resize-none rounded-full bg-transparent px-6 pt-6 pb-4 pl-16 pr-24 text-foreground placeholder:text-muted-foreground/70 focus:outline-none transition-[height,box-shadow] text-lg sm:text-xl leading-relaxed overflow-y-auto font-medium"
            rows={1}
            disabled={isLoading}
          />
          <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    className="h-11 w-11 rounded-full shadow-medium hover:shadow-large transition-all duration-300 hover:scale-105"
                    aria-label="Configurar quantidade de sugestões"
                    title="Configurações"
                    onClick={() => setShowOptions((v) => !v)}
                  >
                    <Settings className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Ajustar quantidade de sugestões
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <Button
              type="submit"
              disabled={!value.trim() || isLoading}
              size="icon"
              className="h-14 w-14 rounded-full bg-gradient-to-br from-primary to-primary/90 text-primary-foreground shadow-large hover:shadow-xl transition-all duration-300 focus-visible:ring-4 focus-visible:ring-offset-2 focus-visible:ring-primary hover:scale-105 active:scale-95"
              aria-label="Enviar busca"
              title="Enviar"
            >
              {isLoading ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                <Send className="h-6 w-6" />
              )}
            </Button>
          </div>
        </div>
      </form>
      
      {/* Search Options */}
      {showOptions && (
        <div className="mt-5 rounded-2xl bg-card/95 border-2 border-border/70 p-6 shadow-xl backdrop-blur-md">
          <div className="flex items-center justify-between">
            <div className="space-y-5 flex-1">
              <div className="text-sm text-muted-foreground font-medium">
                Usando tecnologia de busca semântica (Embeddings + Re‑ranking por IA)
              </div>
              
              <div className="space-y-3">
                <Label htmlFor="topk" className="text-sm font-semibold">
                  Quantidade de sugestões: {topK}
                </Label>
                <input
                  id="topk"
                  type="range"
                  min="1"
                  max="10"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full h-2.5 rounded-lg appearance-none cursor-pointer bg-muted/60 hover:bg-muted/80 transition-colors accent-primary"
                />
                <div className="flex justify-between text-xs text-muted-foreground font-medium">
                  <span>1</span>
                  <span>10</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="mt-7 text-center space-y-3">
        <p className="text-sm text-muted-foreground font-medium">
          💡 <strong className="font-semibold">Dica:</strong> Pressione <kbd className="px-2.5 py-1.5 bg-muted/70 rounded-md text-xs font-mono font-semibold border border-border/60 shadow-sm">Enter</kbd> para quebrar linha, <kbd className="px-2.5 py-1.5 bg-muted/70 rounded-md text-xs font-mono font-semibold border border-border/60 shadow-sm">Ctrl+Enter</kbd> para buscar
        </p>
        <p className="text-xs text-muted-foreground/90 font-medium">
          Separe múltiplos equipamentos por: vírgula (,), ponto-e-vírgula (;) ou quebra de linha
        </p>
      </div>
    </div>
  )
}