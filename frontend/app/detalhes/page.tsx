"use client"

import { useState, useEffect } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { ArrowLeft, FileText, DollarSign, Calendar, Wrench, Building2 } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import { useToast } from "@/hooks/use-toast"
import Image from "next/image"

interface DetailItem {
  fornecedor?: string
  marca?: string
  descricao: string
  valor_unitario?: number
  vida_util_meses?: number
  manutencao?: number
}

interface DetailResponse {
  grupo: string
  items: DetailItem[]
  total: number
}

const API_BASE_URL = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8000'

export default function DetailsPage() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const grupo = searchParams.get('grupo')
  const [details, setDetails] = useState<DetailResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    if (!grupo) {
      router.push('/')
      return
    }

    fetchDetails()
  }, [grupo])

  const fetchDetails = async () => {
    if (!grupo) return

    try {
      const response = await fetch(`${API_BASE_URL}/detalhes/${encodeURIComponent(grupo)}`)
      
      if (!response.ok) {
        throw new Error('Grupo não encontrado')
      }

      const data = await response.json()
      setDetails(data)
    } catch (error) {
      console.error('Erro ao buscar detalhes:', error)
      toast({
        title: "Erro",
        description: "Não foi possível carregar os detalhes",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const currencyBRL = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL', minimumFractionDigits: 2, maximumFractionDigits: 2 })
  const formatPrice = (price: number | undefined) => {
    if (price === null || price === undefined || isNaN(Number(price))) return "N/A"
    return currencyBRL.format(Number(price))
  }

  const formatMaintenance = (maintenance: number | undefined) => {
    if (!maintenance) return "N/A"
    if (maintenance <= 20) return "Baixa"
    if (maintenance <= 50) return "Média"
    return "Alta"
  }

  if (isLoading) {
    return (
      <main className="min-h-screen bg-background">
        <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
          <div className="mb-12 flex justify-between items-center">
            <Button variant="ghost" onClick={() => router.push('/')}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Voltar
            </Button>
            <ThemeToggle />
          </div>
          
          <div className="text-center">
            <div className="inline-flex items-center gap-3 rounded-full bg-primary/10 px-6 py-3 text-primary">
              <div className="h-2 w-2 animate-pulse rounded-full bg-primary"></div>
              <span className="font-medium">Carregando detalhes...</span>
            </div>
          </div>
        </div>
      </main>
    )
  }

  if (!details) {
    return (
      <main className="min-h-screen bg-background">
        <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
          <div className="mb-12 flex justify-between items-center">
            <Button variant="ghost" onClick={() => router.push('/')}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Voltar
            </Button>
            <ThemeToggle />
          </div>
          
          <div className="text-center">
            <h1 className="text-2xl font-bold text-foreground mb-4">
              Detalhes não encontrados
            </h1>
            <p className="text-muted-foreground">
              O grupo solicitado não foi encontrado ou não possui itens.
            </p>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-12 flex justify-between items-center">
          <Button variant="ghost" onClick={() => router.push('/')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Voltar à Busca
          </Button>
          <ThemeToggle />
        </div>

        {/* Title Section */}
        <div className="mb-8 text-center">
          <div className="flex justify-center mb-6">
            <Image
              src="/logo-atlas-horizontal-azul.png"
              alt="Atlas Inovações"
              width={240}
              height={67}
              className="transition-all duration-500 hover:scale-105"
            />
          </div>
          
          <h1 className="text-4xl font-bold tracking-tight mb-4 bg-gradient-to-br from-foreground to-foreground/70 bg-clip-text text-transparent">
            Detalhes do Equipamento
          </h1>
          
          <div className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/20 px-6 py-2 text-primary">
            <FileText className="h-4 w-4" />
            <span className="font-medium">{details.grupo}</span>
          </div>
          
          <p className="text-muted-foreground mt-4">
            {details.total} {details.total === 1 ? 'item encontrado' : 'itens encontrados'}
          </p>
        </div>

        {/* Items Grid */}
        <div className="grid gap-6 lg:grid-cols-2">
          {details.items.map((item, index) => (
            <div
              key={index}
              className="rounded-2xl border border-border bg-card p-6 shadow-lg transition-all duration-300 hover:shadow-xl hover:border-primary/50"
            >
              {/* Item Header */}
              <div className="mb-4 pb-4 border-b border-border">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <FileText className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-lg leading-tight mb-1">
                      {item.descricao}
                    </h3>
                    {(item.fornecedor || item.marca) && (
                      <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                        {item.fornecedor && (
                          <span className="inline-flex items-center gap-1"><Building2 className="h-4 w-4" /> {item.fornecedor}</span>
                        )}
                        {item.marca && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border border-border bg-muted/30 text-foreground/80">Marca: {item.marca}</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Item Stats */}
              <div className="space-y-4">
                {/* Price */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-gradient-to-r from-primary/5 to-transparent border border-primary/20">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <DollarSign className="h-4 w-4 text-primary" />
                    </div>
                    <span className="font-medium">Valor Unitário</span>
                  </div>
                  <span className="font-bold text-primary">
                    {formatPrice(item.valor_unitario)}
                  </span>
                </div>

                {/* Useful Life */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-gradient-to-r from-blue-500/5 to-transparent border border-blue-500/20">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-blue-500/10">
                      <Calendar className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                    </div>
                    <span className="font-medium">Vida Útil</span>
                  </div>
                  <span className="font-bold text-blue-600 dark:text-blue-400">
                    {item.vida_util_meses ? `${item.vida_util_meses} meses` : "N/A"}
                  </span>
                </div>

                {/* Maintenance */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-gradient-to-r from-orange-500/5 to-transparent border border-orange-500/20">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-orange-500/10">
                      <Wrench className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                    </div>
                    <span className="font-medium">Manutenção</span>
                  </div>
                  <span className="font-bold text-orange-600 dark:text-orange-400">
                    {formatMaintenance(item.manutencao)}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Summary Statistics */}
        {details.items.length > 1 && (
          <div className="mt-12 rounded-2xl border border-border bg-gradient-to-br from-card to-card/80 p-8">
            <h2 className="text-2xl font-bold mb-6 text-center">Estatísticas do Grupo</h2>
            
            <div className="grid gap-6 md:grid-cols-3">
              {/* Average Price */}
              <div className="text-center p-4 rounded-xl bg-primary/10 border border-primary/20">
                <div className="flex justify-center mb-2">
                  <div className="p-3 rounded-full bg-primary/20">
                    <DollarSign className="h-6 w-6 text-primary" />
                  </div>
                </div>
                <h3 className="font-semibold mb-1">Preço Médio</h3>
                <p className="text-2xl font-bold text-primary">
                  {(() => {
                    const prices = details.items
                      .map(item => item.valor_unitario)
                      .filter((price): price is number => price !== undefined)
                    const avg = prices.length > 0 
                      ? prices.reduce((sum, price) => sum + price, 0) / prices.length 
                      : 0
                    return formatPrice(avg)
                  })()}
                </p>
              </div>

              {/* Average Life */}
              <div className="text-center p-4 rounded-xl bg-blue-500/10 border border-blue-500/20">
                <div className="flex justify-center mb-2">
                  <div className="p-3 rounded-full bg-blue-500/20">
                    <Calendar className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  </div>
                </div>
                <h3 className="font-semibold mb-1">Vida Útil Média</h3>
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {(() => {
                    const lives = details.items
                      .map(item => item.vida_util_meses)
                      .filter((life): life is number => life !== undefined)
                    const avg = lives.length > 0 
                      ? Math.round(lives.reduce((sum, life) => sum + life, 0) / lives.length)
                      : 0
                    return avg > 0 ? `${avg} meses` : "N/A"
                  })()}
                </p>
              </div>

              {/* Item Count */}
              <div className="text-center p-4 rounded-xl bg-green-500/10 border border-green-500/20">
                <div className="flex justify-center mb-2">
                  <div className="p-3 rounded-full bg-green-500/20">
                    <FileText className="h-6 w-6 text-green-600 dark:text-green-400" />
                  </div>
                </div>
                <h3 className="font-semibold mb-1">Total de Itens</h3>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {details.total}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}