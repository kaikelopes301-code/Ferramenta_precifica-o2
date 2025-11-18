"use client"

import { useEffect, useRef, useState } from "react"
import Link from "next/link"
import dynamic from "next/dynamic"
import { SearchInput } from "@/components/search-input"
import { EquipmentCard } from "@/components/equipment-card"
import { ThemeToggle } from "@/components/theme-toggle"
import { Sparkles, Upload, Download, TrendingUp, Zap, Shield } from "lucide-react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"

// Adia carregamento de componentes não críticos para reduzir JS inicial
const HorizontalScroll = dynamic(() => import("@/components/horizontal-scroll").then(m => m.HorizontalScroll), { ssr: false, loading: () => null })
const CartWidget = dynamic(() => import("@/components/cart-widget").then(m => m.CartWidget), { ssr: false, loading: () => null })
type CartItem = import("@/components/cart-widget").CartItem

export type Equipment = {
  ranking: number
  sugeridos: string
  valor_unitario: number | null
  vida_util_meses: number | null
  manutencao_percent: number | null
  confianca: number | null
  link_detalhes: string
  isIncorrect?: boolean
  feedback?: string
  equipamento_material_revisado?: string
  marca?: string | null
  origemDescricao?: string | null
}

const API_BASE_URL = '/api'

export default function Home() {
  const [equipments, setEquipments] = useState<Equipment[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [hasData, setHasData] = useState(false)
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single')
  const [uploadingFile, setUploadingFile] = useState(false)
  const [batchResults, setBatchResults] = useState<any[]>([])
  const [batchGroups, setBatchGroups] = useState<Array<{ descricao: string; itens: Equipment[] }>>([])
  const [lastQuery, setLastQuery] = useState("")
  const [batchSortMap, setBatchSortMap] = useState<Record<string, string>>({})
  const { toast } = useToast()
  const resultsRef = useRef<HTMLDivElement | null>(null)
  const USER_ID = 'demo-user'
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [lastSelectedIdx, setLastSelectedIdx] = useState<number | null>(null)
  const [cart, setCart] = useState<CartItem[]>([])
  const [singleFilter, setSingleFilter] = useState("")
  const [singleSort, setSingleSort] = useState<'conf-desc'|'price-asc'|'price-desc'|'life-desc'>('conf-desc')
  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    checkDataStatus()
  }, [])

  useEffect(() => {
    if (typeof document !== 'undefined') {
      setIsDark(document.documentElement.classList.contains('dark'))
    }
    const handler = (e: any) => {
      const theme = e?.detail?.theme
      if (theme === 'dark') setIsDark(true)
      else if (theme === 'light') setIsDark(false)
      else if (typeof document !== 'undefined') setIsDark(document.documentElement.classList.contains('dark'))
    }
    window.addEventListener('theme-changed', handler as EventListener)
    return () => window.removeEventListener('theme-changed', handler as EventListener)
  }, [])

  const checkDataStatus = async () => {
    try {
      const response = await fetch(`/api/data/status`)
      const data = await response.json()
      setHasData(data.has_data)
    } catch (error) {
      console.error('Erro ao verificar status dos dados:', error)
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploadingFile(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`/api/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Erro no upload')
      }

      const result = await response.json()
      setHasData(true)
      toast({
        title: "✅ Sucesso!",
        description: `Planilha carregada com ${result.rows} linhas`,
      })
    } catch (error) {
      toast({
        title: "❌ Erro",
        description: "Falha ao carregar planilha",
        variant: "destructive",
      })
    } finally {
      setUploadingFile(false)
      event.target.value = ''
    }
  }

  useEffect(() => {
    if (!isLoading && (equipments.length > 0 || batchGroups.length > 0)) {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }, [isLoading, equipments.length, batchGroups.length])

  const splitDescriptions = (text: string): string[] => {
    // Divide por: quebras de linha, ponto-e-vírgula, ou vírgula
    // O lookahead (?!\d) evita quebrar números decimais (1,5) e (?!\s*\d) evita quebrar "volume 1, 2 litros"
    const parts = text
      .split(/\r?\n|;\s*|,\s*(?!\d)/g)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
    return parts.length > 0 ? parts : [text.trim()]
  }

  const itemId = (e: Equipment) => `${e.sugeridos}__${e.valor_unitario ?? 'na'}__${e.vida_util_meses ?? 'na'}`

  const addToCart = (e: Equipment) => {
    const baseId = itemId(e)
    const descKey = (e.origemDescricao ?? lastQuery ?? '').trim()
    const cartId = `${baseId}__d:${descKey}`
    setCart(prev => {
      const idx = prev.findIndex(it => it.id === cartId)
      if (idx >= 0) {
        const copy = [...prev]
        copy[idx] = { ...copy[idx], qty: copy[idx].qty + 1 }
        return copy
      }
      return [...prev, { id: cartId, name: e.sugeridos, price: e.valor_unitario ?? null, qty: 1, vidaUtilMeses: e.vida_util_meses ?? null, manutencaoPercent: e.manutencao_percent ?? null, fornecedor: null, marca: e.marca ?? null, descricao: e.origemDescricao ?? lastQuery }]
    })
    toast({ 
      title: '🛒 Adicionado ao carrinho', 
      description: e.sugeridos,
      // Removed duration as it is not part of ToastProps
    })
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new Event('cart:add'))
    }
  }

  const addSelectedToCart = (list: Equipment[]) => {
    if (selected.size === 0) return
    setCart(prev => {
      const next = [...prev]
      for (const e of list) {
        const baseId = itemId(e)
        if (!selected.has(baseId)) continue
        const descKey = (e.origemDescricao ?? lastQuery ?? '').trim()
        const cartId = `${baseId}__d:${descKey}`
        const idx = next.findIndex(it => it.id === cartId)
        if (idx >= 0) next[idx] = { ...next[idx], qty: next[idx].qty + 1 }
        else next.push({ id: cartId, name: e.sugeridos, price: e.valor_unitario ?? null, qty: 1, vidaUtilMeses: e.vida_util_meses ?? null, manutencaoPercent: e.manutencao_percent ?? null, fornecedor: null, marca: e.marca ?? null, descricao: e.origemDescricao ?? lastQuery })
      }
      return next
    })
    toast({ title: '✅ Itens adicionados', description: `${selected.size} selecionados` })
    setSelected(new Set())
    setLastSelectedIdx(null)
  }

  const clearCart = () => setCart([])
  const removeFromCart = (id: string) => setCart(prev => prev.filter(it => it.id !== id))
  const changeQty = (id: string, qty: number) => setCart(prev => prev.map(it => it.id === id ? { ...it, qty: Math.max(1, qty) } : it))
  const changeNotes = (id: string, notes: string) => setCart(prev => prev.map(it => it.id === id ? { ...it, notes } : it))
  const changeName = (id: string, name: string) => setCart(prev => prev.map(it => it.id === id ? { ...it, name } : it))

  const handleSearch = async (description: string, options: { topK: number, useTfidf: boolean }) => {
    setLastQuery(description)
    if (!hasData) {
      toast({
        title: "⚠️ Dados necessários",
        description: "Faça upload de uma planilha primeiro",
        variant: "destructive",
      })
      return
    }

    setIsLoading(true)
    setEquipments([])
    setBatchResults([])
    setBatchGroups([])

    try {
      const descricoes = splitDescriptions(description)

      if (descricoes.length === 1) {
        // Busca individual (1 descrição apenas)
        const searchResponse = await fetch(`/api/smart-search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-user-id': USER_ID },
          body: JSON.stringify({ q: descricoes[0] || description, top_k: options.topK })
        })
        // Tolera resposta não-OK: tenta extrair JSON e faz fallback gracioso
        const searchData = await searchResponse.json().catch(() => ({ resultados: [] }))
        if (!searchResponse.ok) {
          console.warn('Busca inteligente retornou erro', searchData)
          toast({
            title: '⚠️ Aviso',
            description: 'O serviço de busca demorou ou retornou erro. Tentaremos novamente em seguida.',
          })
        }
        const mapped = (searchData.resultados || []).map((r:any, idx: number) => ({
          ranking: r.ranking ?? (idx + 1),
          sugeridos: r.sugeridos,
          valor_unitario: r.valor_unitario ?? null,
          vida_util_meses: r.vida_util_meses ?? null,
          manutencao_percent: r.manutencao_percent ?? null,
          confianca: r.score ?? r.confianca ?? null,
          link_detalhes: r.link_detalhes || '#',
          marca: r.marca ?? null,
          origemDescricao: descricoes[0] || description
        }))
        setEquipments(mapped)
      } else {
        // Busca em lote (2+ descrições)
        const resp = await fetch(`/api/smart-search/batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-user-id': USER_ID },
          body: JSON.stringify({
            descricoes,
            top_k: options.topK,
            use_tfidf: options.useTfidf,
          }),
        })
        // Tolerar não-OK e continuar com fallback
        const data = await resp.json().catch(() => ({ resultados: [] }))
        if (!resp.ok) {
          console.warn('Busca em lote retornou erro', data)
          toast({
            title: '⚠️ Aviso',
            description: 'A busca em lote encontrou um erro. Resultados parciais podem estar vazios.',
          })
        }
        const rows = (data.resultados || []) as Array<any>
        setBatchResults(rows)
        const map: Record<string, Equipment[]> = {}
        for (const r of rows) {
          // CORRIGIDO: Backend retorna 'query_original', não 'descricao_original'
          const desc = (r.query_original || r.descricao_original) as string
          if (!map[desc]) map[desc] = []
          map[desc].push({
            ranking: 0,
            sugeridos: r.sugerido || r.sugeridos || 'N/A',
            valor_unitario: r.valor_unitario ?? null,
            vida_util_meses: r.vida_util_meses ?? null,
            manutencao_percent: r.manutencao_percent ?? null,
            confianca: r.confianca ?? r.score ?? null,
            link_detalhes: r.link_detalhes || '#',
            marca: r.marca ?? null,
            origemDescricao: desc,
          })
        }
        const groups: Array<{ descricao: string; itens: Equipment[] }> = []
        Object.entries(map).forEach(([descricao, itens]) => {
          const ordered = itens.sort((a, b) => (b.confianca ?? 0) - (a.confianca ?? 0))
          ordered.forEach((it, idx) => (it.ranking = idx + 1))
          groups.push({ descricao, itens: ordered })
        })
        const orderedGroups = descricoes
          .filter(d => groups.find(g => g.descricao === d))
          .map(d => groups.find(g => g.descricao === d)!)
        setBatchGroups(orderedGroups)
      }

    } catch (error) {
      console.error('Erro na busca:', error)
      toast({
        title: "❌ Erro",
        description: "Falha na busca. Tente .",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="app-container section-spacing">
        {/* Header com gradiente melhorado */}
        <div className="mb-20 px-2 sm:px-0 relative overflow-hidden">
          <div className="hidden lg:block absolute inset-0 -z-10 gradient-mesh" />
          <div className="hidden lg:block absolute inset-0 -z-10 bg-gradient-to-b from-primary/5 via-transparent to-transparent" />
          
          {/* Top bar */}
          <div className="mb-12 flex items-center justify-between animate-fade-slide-up">
            <Link 
              href="/" 
              className="flex items-center gap-2 transition-all duration-300 hover:scale-[1.02] active:scale-[0.98]" 
              aria-label="Ir para a página inicial"
            >
              <Image
                src={isDark ? "/logo-atlas-branca.png" : "/logo-atlas-letras-preta.png"}
                alt="Atlas Inovações"
                width={260}
                height={76}
                priority
                className="select-none drop-shadow-sm"
              />
            </Link>
            <div className="transition-all duration-300 hover:scale-110 active:scale-95">
              <ThemeToggle />
            </div>
          </div>
          
          <div className="text-center space-y-10">
            <div className="space-y-7 animate-fade-slide-up" style={{ animationDelay: '100ms' }}>
              <div className="inline-flex items-center gap-2.5 rounded-full bg-gradient-to-r from-primary/20 via-primary/15 to-primary/10 border border-primary/40 px-6 py-2.5 text-sm text-primary backdrop-blur-sm shadow-medium hover:shadow-large transition-all duration-300 hover:scale-[1.02]">
                <Sparkles className="h-4 w-4 animate-pulse-glow" />
                <span className="font-semibold tracking-wide">Ferramenta de Precificação Inteligente</span>
              </div>
              
              <h1 className="mb-8 text-balance font-bold tracking-tight fluid-h1 bg-gradient-to-br from-foreground via-foreground/95 to-foreground/80 bg-clip-text text-transparent drop-shadow-sm">
                Precificação de Equipamentos
              </h1>
              
              <p className="mx-auto max-w-3xl text-pretty text-muted-foreground leading-relaxed fluid-subtitle font-medium">
                Descreva os equipamentos que precisa e receba sugestões com preço, vida útil e manutenção — tudo com IA.
              </p>
            </div>
            
            {/* Feature badges */}
            <div className="flex flex-wrap justify-center gap-4 animate-fade-slide-up" style={{ animationDelay: '200ms' }}>
              <div className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-card/70 border border-border/60 backdrop-blur-md shadow-medium hover:shadow-large transition-all duration-300 hover:scale-[1.05] hover:border-primary/40">
                <Zap className="h-4 w-4 text-primary" />
                <span className="text-sm font-semibold text-muted-foreground">Busca Instantânea</span>
              </div>
              <div className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-card/70 border border-border/60 backdrop-blur-md shadow-medium hover:shadow-large transition-all duration-300 hover:scale-[1.05] hover:border-primary/40">
                <TrendingUp className="h-4 w-4 text-primary" />
                <span className="text-sm font-semibold text-muted-foreground">Análise Inteligente</span>
              </div>
              <div className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-card/70 border border-border/60 backdrop-blur-md shadow-medium hover:shadow-large transition-all duration-300 hover:scale-[1.05] hover:border-primary/40">
                <Shield className="h-4 w-4 text-primary" />
                <span className="text-sm font-semibold text-muted-foreground">Dados Confiáveis</span>
              </div>
            </div>
          </div>
        </div>

        {/* Upload Section melhorado */}
        {!hasData && (
          <div className="mb-20 animate-fade-slide-up" style={{ animationDelay: '300ms' }}>
            <div className="rounded-3xl border-2 border-dashed border-border/70 bg-gradient-to-br from-card via-card/98 to-card/95 p-14 text-center shadow-large hover:shadow-xl hover:border-primary/50 transition-all duration-500 card-glass">
              <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-primary/25 via-primary/20 to-primary/15 mb-7 shadow-large relative">
                <div className="absolute inset-0 rounded-full bg-primary/15 animate-ping opacity-75"></div>
                <Upload className="h-11 w-11 text-primary relative z-10" />
              </div>
              <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-foreground to-foreground/85 bg-clip-text text-transparent">
                Carregue sua planilha
              </h3>
              <p className="text-muted-foreground mb-10 text-lg max-w-md mx-auto leading-relaxed">
                Faça upload de um arquivo Excel (.xlsx) com os dados dos equipamentos
              </p>
              <div className="flex flex-col items-center gap-5">
                <Label htmlFor="file-upload" className="cursor-pointer">
                  <Button 
                    asChild 
                    disabled={uploadingFile}
                    size="lg"
                    className="btn-interactive shadow-large hover:shadow-xl px-10 py-7 text-base font-bold tracking-wide"
                  >
                    <span className="flex items-center gap-3">
                      {uploadingFile ? (
                        <>
                          <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent"></div>
                          Carregando...
                        </>
                      ) : (
                        <>
                          <Upload className="h-5 w-5" />
                          Selecionar arquivo
                        </>
                      )}
                    </span>
                  </Button>
                </Label>
                <Input
                  id="file-upload"
                  type="file"
                  accept=".xlsx"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <p className="text-xs text-muted-foreground font-medium">Formatos aceitos: .xlsx (máx. 10MB)</p>
              </div>
            </div>
          </div>
        )}

        {/* Search Input */}
        {hasData && (
          <div className="space-y-8 animate-fade-slide-up" style={{ animationDelay: '200ms' }}>
            <SearchInput onSearch={handleSearch} isLoading={isLoading} />
          </div>
        )}

        {/* Results */}
        <div ref={resultsRef} />
        
        {/* Loading State melhorado */}
        {isLoading && (
          <div className="mt-20">
            <div className="mb-12 text-center animate-fade-slide-up">
              <div className="inline-flex items-center gap-4 rounded-full bg-gradient-to-r from-primary/15 via-primary/20 to-primary/15 px-10 py-5 text-primary shadow-large backdrop-blur-md border border-primary/30">
                <div className="relative">
                  <div className="h-4 w-4 animate-ping rounded-full bg-primary absolute"></div>
                  <div className="h-4 w-4 rounded-full bg-primary"></div>
                </div>
                <span className="font-bold text-lg tracking-wide">Analisando equipamentos...</span>
              </div>
            </div>
            <div className="grid gap-8 grid-cols-[repeat(auto-fit,minmax(280px,1fr))] 2xl:grid-cols-[repeat(auto-fit,minmax(300px,1fr))]">
              {[1, 2, 3].map((i) => (
                <div key={i} className="group">
                  <div className="h-96 rounded-3xl bg-gradient-to-br from-card to-card/60 border border-border/60 shadow-large relative overflow-hidden">
                    <div className="absolute inset-0 animate-shimmer"></div>
                  </div>
                  <div className="mt-6 space-y-3">
                    <div className="h-6 rounded-lg bg-muted/60 skeleton w-3/4"></div>
                    <div className="h-4 rounded-lg bg-muted/40 skeleton w-1/2"></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results - Single */}
        {!isLoading && equipments.length > 0 && batchGroups.length === 0 && (
          <div className="mt-20 animate-fade-slide-up">
            <div className="mb-12 flex items-center justify-between flex-wrap gap-6">
              <div>
                <h2 className="text-4xl font-black tracking-tight bg-gradient-to-r from-foreground to-foreground/85 bg-clip-text text-transparent">
                  Sugestões de Equipamentos
                </h2>
                <p className="text-muted-foreground mt-3 text-lg font-medium">Resultados baseados na sua descrição</p>
              </div>
              <div className="flex items-center gap-3">
                <span className="inline-flex items-center gap-3 rounded-full bg-gradient-to-r from-primary/20 via-primary/15 to-primary/10 px-6 py-3 text-sm font-bold text-primary border border-primary/30 shadow-medium">
                  <span className="flex h-3 w-3 rounded-full bg-primary shadow-[0_0_10px_rgba(var(--primary),0.7)]"></span>
                  {equipments.length} {equipments.length === 1 ? "resultado" : "resultados"}
                </span>
              </div>
            </div>
            <HorizontalScroll itemMinWidth={240}>
              {equipments.map((equipment, index) => (
                <EquipmentCard
                  key={equipment.ranking}
                  equipment={equipment}
                  dense
                  selected={selected.has(itemId(equipment))}
                  onToggleSelect={() => {
                    const id = itemId(equipment)
                    setSelected(prev => { const next = new Set(prev); if(next.has(id)) next.delete(id); else next.add(id); return next })
                  }}
                  onAdd={() => addToCart(equipment)}
                />
              ))}
            </HorizontalScroll>
          </div>
        )}

        {/* Results - Batch */}
        {!isLoading && batchGroups.length > 0 && (
          <div className="mt-20 space-y-16 animate-fade-slide-up">
            <div className="flex items-center justify-end gap-3">
              <Button 
                size="sm" 
                variant="outline" 
                className="btn-interactive shadow-soft hover:shadow-medium"
                onClick={() => {
                  try {
                    const headers = [
                      'Descrição Original',
                      'Equipamento Sugerido',
                      'Valor Unitário',
                      'Vida Útil (meses)',
                      'Manutenção (%)',
                      'Confiança (%)',
                      'Marca',
                    ]
                    const rows = (batchResults || []).map((r:any) => [
                      `"${r.descricao_original || ''}"`,
                      `"${r.sugerido || 'N/A'}"`,
                      r.valor_unitario ?? 'N/A',
                      r.vida_util_meses ?? 'N/A',
                      r.manutencao_percent ?? 'N/A',
                      r.confianca ?? 'N/A',
                      `"${r.marca || ''}"`,
                    ].join(','))
                    const csv = [headers.join(','), ...rows].join('\n')
                    const blob = new Blob([csv], { type: 'text/csv' })
                    const url = window.URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = 'resultados_busca_lote.csv'
                    a.click()
                    window.URL.revokeObjectURL(url)
                    toast({ title: '✅ Download iniciado', description: 'CSV gerado com sucesso' })
                  } catch (e) {
                    console.error(e)
                    toast({ title: '❌ Erro', description: 'Falha ao gerar CSV', variant: 'destructive' })
                  }
                }}
              >
                <Download className="mr-2 h-4 w-4" />
                Baixar CSV
              </Button>
            </div>
            
            {batchGroups.map((group, gi) => {
              const sortKey = batchSortMap[group.descricao] || 'conf-desc'
              const sortedItems = [...group.itens].sort((a, b) => {
                switch (sortKey) {
                  case 'price-asc':
                    return (a.valor_unitario ?? Infinity) - (b.valor_unitario ?? Infinity)
                  case 'price-desc':
                    return (b.valor_unitario ?? -Infinity) - (a.valor_unitario ?? -Infinity)
                  case 'life-desc':
                    return (b.vida_util_meses ?? 0) - (a.vida_util_meses ?? 0)
                  case 'conf-asc':
                    return (a.confianca ?? 0) - (b.confianca ?? 0)
                  case 'conf-desc':
                  default:
                    return (b.confianca ?? 0) - (a.confianca ?? 0)
                }
              })
              sortedItems.forEach((it, idx) => (it.ranking = idx + 1))
              
              return (
                <section key={gi} className="space-y-8">
                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <div>
                      <h2 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent">
                        Resultados para: "{group.descricao}"
                      </h2>
                      <p className="text-muted-foreground mt-2 text-lg">
                        {group.itens.length} {group.itens.length === 1 ? 'sugestão' : 'sugestões'}
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      <label htmlFor={`sort-${gi}`} className="text-sm font-medium text-muted-foreground">
                        Ordenar por
                      </label>
                      <select
                        id={`sort-${gi}`}
                        value={sortKey}
                        onChange={(e) => setBatchSortMap((m) => ({ ...m, [group.descricao]: e.target.value }))}
                        className="px-4 py-2 rounded-lg border border-border bg-card/50 text-sm font-medium shadow-soft hover:shadow-medium focus-ring transition-all backdrop-blur-sm"
                      >
                        <option value="conf-desc">Maior confiança</option>
                        <option value="conf-asc">Menor confiança</option>
                        <option value="price-asc">Menor preço</option>
                        <option value="price-desc">Maior preço</option>
                        <option value="life-desc">Maior vida útil</option>
                      </select>
                    </div>
                  </div>
                  <HorizontalScroll itemMinWidth={230}>
                    {sortedItems.map((equipment, index) => (
                      <EquipmentCard
                        key={`${group.descricao}-${equipment.sugeridos}-${index}`}
                        equipment={equipment}
                        dense
                        selected={selected.has(itemId(equipment))}
                        onToggleSelect={() => {
                          const id = itemId(equipment)
                          setSelected(prev => { const next = new Set(prev); if(next.has(id)) next.delete(id); else next.add(id); return next })
                        }}
                        onAdd={() => addToCart(equipment)}
                      />
                    ))}
                  </HorizontalScroll>
                </section>
              )
            })}
          </div>
        )}

        {/* Empty State melhorado */}
        {!isLoading && hasData && equipments.length === 0 && batchGroups.length === 0 && (
          <div className="mt-32 text-center animate-fade-slide-up">
            <div className="mx-auto mb-10 flex h-28 w-28 items-center justify-center rounded-full bg-gradient-to-br from-primary/25 via-primary/20 to-primary/15 border-2 border-primary/30 shadow-xl relative">
              <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping"></div>
              <Sparkles className="h-14 w-14 text-primary animate-pulse-glow relative z-10" />
            </div>
            <h3 className="mb-5 text-3xl font-black bg-gradient-to-r from-foreground to-foreground/85 bg-clip-text text-transparent">
              Comece sua busca
            </h3>
            <p className="text-muted-foreground text-xl mb-10 max-w-2xl mx-auto leading-relaxed font-medium">
              Digite a descrição dos equipamentos que você precisa precificar e receba sugestões inteligentes
            </p>
            {lastQuery && (
              <div className="mb-12 max-w-2xl mx-auto text-left card-glass rounded-2xl p-7 shadow-xl border-2 border-border/70">
                <p className="text-sm font-bold text-primary mb-4 flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Sugestões para melhorar sua busca:
                </p>
                <ul className="list-disc pl-6 space-y-2.5 text-sm text-muted-foreground font-medium">
                  {(() => {
                    const tips: string[] = []
                    const q = lastQuery.toLowerCase()
                    if (!/(110|127|220|12v|24v|bivolt|\bv\b)/.test(q)) tips.push('Inclua voltagem ou tensão (ex.: 220V, 127V, bivolt).')
                    if (!/(nylon|aço|inox|pi[aã]cava|algod[aã]o|microfibra|pl[aá]stico|borracha)/.test(q)) tips.push('Informe material (ex.: nylon, inox, microfibra, piaçava).')
                    if (!/(tamanho|\b\d+\s?cm\b|\b\d+\s?mm\b|\b\d+\s?l\b)/.test(q)) tips.push('Adicione tamanho/capacidade (ex.: 60 cm, 10 mm, 20 L).')
                    if (!/(bosch|makita|karcher|wap|3m|tramontina|voith|flash\s?limp|kärcher)/.test(q) && !/marca|modelo/.test(q)) tips.push('Se souber, inclua marca/modelo para maior precisão.')
                    if (tips.length === 0) tips.push('Sua descrição já está bem detalhada! ✨')
                    return tips.map((t, i) => <li key={i}>{t}</li>)
                  })()}
                </ul>
              </div>
            )}
            <div className="flex flex-wrap justify-center gap-3 text-sm">
              {['vassouras', 'mops', 'aspiradores', 'panos', 'baldes'].map((tag) => (
                <span 
                  key={tag}
                  className="px-5 py-2.5 rounded-full bg-muted/60 text-muted-foreground hover:bg-primary/15 hover:text-primary transition-all duration-300 cursor-pointer hover:scale-105 shadow-medium hover:shadow-large font-semibold"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Future AI Assistant melhorado */}
        <div className="mt-32 relative animate-fade-slide-up">
          <div className="rounded-3xl border-2 border-border/70 bg-gradient-to-br from-card via-card/98 to-card/95 p-14 text-center backdrop-blur-md shadow-xl overflow-hidden card-glass">
            <div className="absolute inset-0 gradient-mesh"></div>
            <div className="absolute top-0 right-0 w-48 h-48 bg-primary/15 rounded-full blur-3xl -translate-y-24 translate-x-24"></div>
            <div className="absolute bottom-0 left-0 w-40 h-40 bg-secondary/15 rounded-full blur-3xl translate-y-20 -translate-x-20"></div>
            
            <div className="relative z-10">
              <div className="mx-auto mb-10 flex h-24 w-24 items-center justify-center rounded-full bg-gradient-to-br from-primary via-primary/95 to-primary/85 shadow-xl relative">
                <div className="absolute inset-0 rounded-full bg-primary/40 animate-ping"></div>
                <Sparkles className="h-12 w-12 text-primary-foreground animate-pulse-glow relative z-10" />
              </div>
              <h3 className="mb-6 text-3xl font-black bg-gradient-to-r from-foreground via-primary/90 to-foreground bg-clip-text text-transparent">
                Assistente Inteligente em Breve
              </h3>
              <p className="text-muted-foreground text-lg leading-relaxed max-w-2xl mx-auto mb-10 font-medium">
                Em breve você poderá conversar com nosso chatbot powered by IA para receber sugestões personalizadas de equipamentos com
                melhor performance, preços competitivos e análises detalhadas.
              </p>
              <div className="flex flex-wrap justify-center gap-4 text-sm">
                <div className="flex items-center gap-2 px-6 py-3.5 rounded-full bg-primary/20 text-primary border-2 border-primary/30 shadow-medium hover:shadow-xl transition-all hover:scale-105 font-bold">
                  <div className="h-2.5 w-2.5 rounded-full bg-primary shadow-[0_0_8px_rgba(var(--primary),0.7)]"></div>
                  <span>Análise Inteligente</span>
                </div>
                <div className="flex items-center gap-2 px-6 py-3.5 rounded-full bg-secondary/60 text-secondary-foreground border-2 border-secondary/30 shadow-medium hover:shadow-xl transition-all hover:scale-105 font-bold">
                  <div className="h-2.5 w-2.5 rounded-full bg-secondary-foreground/70"></div>
                  <span>Recomendações Personalizadas</span>
                </div>
                <div className="flex items-center gap-2 px-6 py-3.5 rounded-full bg-accent/20 text-accent-foreground border-2 border-accent/30 shadow-medium hover:shadow-xl transition-all hover:scale-105 font-bold">
                  <div className="h-2.5 w-2.5 rounded-full bg-accent"></div>
                  <span>Comparação de Preços</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer melhorado */}
      <footer className="border-t border-border/60 bg-gradient-to-b from-card/60 to-card/40 backdrop-blur-md mt-32">
        <div className="app-container py-12">
          <div className="flex flex-col items-center justify-between gap-8 sm:flex-row">
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <span className="font-semibold">© 2025 Atlas Inovações</span>
              <span className="text-border">•</span>
              <span className="font-medium">Ferramenta de Precificação Inteligente</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="flex items-center gap-2 font-semibold">
                Powered by <Sparkles className="h-4 w-4 text-primary animate-pulse-glow" /> IA
              </span>
            </div>
          </div>
        </div>
      </footer>

      <CartWidget
        items={cart}
        onClear={clearCart}
        onRemove={removeFromCart}
        onChangeQty={changeQty}
        onChangeNotes={changeNotes}
        onChangeName={changeName}
      />
    </main>
  )
}
