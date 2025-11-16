"use client"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Wrench, DollarSign, Calendar, TrendingUp, CheckCircle2, ShoppingCart, Info, X } from "lucide-react"
import type { Equipment } from "@/app/page"
import { useState } from "react"
import * as Dialog from "@radix-ui/react-dialog"

interface EquipmentCardProps {
  equipment: Equipment
  dense?: boolean
  selected?: boolean
  onToggleSelect?: (equipment: Equipment) => void
  onAdd?: (equipment: Equipment) => void
}

export function EquipmentCard({ equipment, dense, selected = false, onToggleSelect, onAdd }: EquipmentCardProps) {
  const [isAdding, setIsAdding] = useState(false)
  const [isDetailsOpen, setIsDetailsOpen] = useState(false)

  const getConfidenceConfig = (confidence: number | null) => {
    if (!confidence) return { 
      color: "text-muted-foreground", 
      bg: "bg-muted/10",
      label: "N/A",
      icon: "⚪"
    }
    if (confidence >= 90) return { 
      color: "text-emerald-600 dark:text-emerald-400", 
      bg: "bg-emerald-500/10",
      label: "Excelente",
      icon: "🟢"
    }
    if (confidence >= 75) return { 
      color: "text-yellow-600 dark:text-yellow-400", 
      bg: "bg-yellow-500/10",
      label: "Boa",
      icon: "🟡"
    }
    return { 
      color: "text-orange-600 dark:text-orange-400", 
      bg: "bg-orange-500/10",
      label: "Moderada",
      icon: "🟠"
    }
  }

  const getMaintenanceConfig = (maintenance: number | null) => {
    if (!maintenance || maintenance === 0) return {
      bg: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30",
      label: "Sem Manutenção",
     
      description: "Não requer manutenção"
    }
    if (maintenance <= 20) {
      return {
        bg: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30",
        label: "Baixa",
       
        description: "Manutenção mínima"
      }
    }
    if (maintenance <= 50) {
      return {
        bg: "bg-yellow-500/15 text-yellow-700 dark:text-yellow-400 border-yellow-500/30",
        label: "Média",
        icon: "⚠️",
        description: "Manutenção moderada"
      }
    }
    return {
      bg: "bg-orange-500/15 text-orange-700 dark:text-orange-400 border-orange-500/30",
      label: "Alta",
    
      description: "Manutenção frequente"
    }
  }

  const currencyBRL = new Intl.NumberFormat('pt-BR', { 
    style: 'currency', 
    currency: 'BRL', 
    minimumFractionDigits: 2, 
    maximumFractionDigits: 2 
  })

  const formatPrice = (price: number | null) => {
    if (price === null || price === undefined || isNaN(Number(price))) return "Não informado"
    return currencyBRL.format(Number(price))
  }

  const confidenceConfig = getConfidenceConfig(equipment.confianca)
  const maintenanceConfig = getMaintenanceConfig(equipment.manutencao_percent)

  const handleAdd = async () => {
    setIsAdding(true)
    onAdd?.(equipment)
    // Animação visual de feedback
    setTimeout(() => setIsAdding(false), 600)
  }

  return (
    <Card
      className={`group relative overflow-hidden border-border/60 bg-gradient-to-br from-card via-card/99 to-card/97 shadow-md transition-all duration-300 hover:border-primary/50 hover:shadow-lg hover:-translate-y-1 focus-within:ring-2 focus-within:ring-primary/25 flex flex-col ${
        selected ? 'ring-2 ring-primary/40 border-primary/60 shadow-lg' : ''
      } ${dense ? 'w-[300px] h-[450px]' : 'w-[320px] h-[490px]'}`}
    >
      {/* Gradiente decorativo no topo */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary/0 via-primary/60 to-primary/0"></div>
      
      {/* Badge de seleção */}
      {selected && (
        <div className="absolute top-2.5 left-2.5 z-10 animate-pop-in">
          <Badge className="bg-primary/90 text-primary-foreground border-primary shadow-lg backdrop-blur-sm font-semibold text-xs">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            Selecionado
          </Badge>
        </div>
      )}

      <CardHeader className={`${dense ? 'pb-1 pt-3.5' : 'pb-1.5 pt-4'} relative flex-shrink-0`}>
        <div className="flex items-start justify-between gap-2.5">
          <div className="flex-1 min-w-0 space-y-2">
            {/* Ranking badge */}
            <div className="flex items-center gap-1.5 flex-wrap">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-gradient-to-r from-primary/15 to-primary/10 border border-primary/35 shadow-sm">
                <TrendingUp className="h-3 w-3 text-primary flex-shrink-0" />
                <span className="text-xs font-extrabold text-primary">#{equipment.ranking}</span>
              </div>
              {equipment.confianca && equipment.confianca >= 90 && (
                <Badge variant="outline" className="rounded-full text-[10px] px-2 py-0.5 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/35 font-bold">
                  Top
                </Badge>
              )}
            </div>

            {/* Nome do equipamento - altura fixa com truncamento */}
            <div className={`${dense ? 'h-[40px]' : 'h-[44px]'} overflow-hidden`}>
              <h3 
                className={`${dense ? 'text-base' : 'text-lg'} font-bold leading-tight text-foreground group-hover:text-primary transition-colors line-clamp-2`}
                title={equipment.sugeridos}
              >
                {equipment.sugeridos}
              </h3>
            </div>

            {/* Marca - altura fixa */}
            <div className="h-[24px] flex items-center">
              {equipment.marca && (
                <Badge variant="outline" className="rounded-full text-[11px] px-2.5 py-0.5 bg-secondary/50 border-border/50 font-semibold truncate max-w-full">
                  {equipment.marca}
                </Badge>
              )}
            </div>
          </div>

          {/* Checkbox */}
          <div className="flex flex-col items-center gap-1 pt-0.5 flex-shrink-0">
            <label className="relative inline-flex items-center cursor-pointer group/checkbox">
              <input
                type="checkbox"
                aria-label="Selecionar sugestão"
                className="sr-only peer"
                checked={selected}
                onChange={() => onToggleSelect?.(equipment)}
              />
              <div className="w-5 h-5 rounded-md border-2 border-border peer-checked:border-primary peer-checked:bg-primary transition-all duration-200 flex items-center justify-center group-hover/checkbox:border-primary/50 group-hover/checkbox:scale-110">
                {selected && (
                  <CheckCircle2 className="h-3.5 w-3.5 text-primary-foreground animate-pop-in" />
                )}
              </div>
            </label>
          </div>
        </div>
      </CardHeader>

      <CardContent className={`${dense ? 'pb-3 px-4' : 'pb-3.5 px-5'} flex-1 min-h-0`}>
        <div className={`${dense ? 'space-y-3' : 'space-y-3.5'}`}>
          {/* Preço destacado - altura fixa */}
          <div className={`relative overflow-hidden rounded-xl bg-gradient-to-br from-primary/8 via-primary/12 to-primary/6 ${dense ? 'p-3.5' : 'p-4'} border border-primary/25 shadow-md hover:shadow-lg transition-all duration-300 group/price`}>
            <div className="flex items-center gap-3">
              <div className={`flex ${dense ? 'h-11 w-11' : 'h-14 w-14'} items-center justify-center rounded-xl bg-gradient-to-br from-primary via-primary/95 to-primary/90 shadow-lg group-hover/price:shadow-xl transition-all duration-300 group-hover/price:scale-105 flex-shrink-0`}>
                <DollarSign className={`${dense ? 'h-5 w-5' : 'h-7 w-7'} text-primary-foreground`} />
              </div>
              <div className="flex-1 min-w-0 space-y-2">
                <div>
                  <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-1 truncate">Valor Unitário</p>
                  <p className={`${dense ? 'text-lg' : 'text-xl'} font-extrabold text-primary leading-none truncate`}>
                    {formatPrice(equipment.valor_unitario)}
                  </p>
                </div>
                {/* Valor mensal calculado */}
                {equipment.valor_unitario && equipment.vida_util_meses && equipment.vida_util_meses > 0 && (
                  <div className="pt-2 border-t border-primary/20">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[9px] font-semibold text-muted-foreground uppercase tracking-wide truncate">Custo Mensal</p>
                      <p className="text-sm font-bold text-primary/80 flex-shrink-0">
                        {formatPrice(equipment.valor_unitario / equipment.vida_util_meses)}
                        <span className="text-[10px] font-medium text-muted-foreground ml-1">/mês</span>
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Grid de estatísticas - altura fixa */}
          <div className={`grid grid-cols-2 ${dense ? 'gap-2.5' : 'gap-3'}`}>
            {/* Vida útil */}
            <div className={`group/stat rounded-lg border border-border/50 bg-gradient-to-br from-blue-500/5 to-blue-500/8 ${dense ? 'p-2.5' : 'p-3'} hover:shadow-sm hover:border-blue-500/25 transition-all duration-300 hover:-translate-y-0.5 flex flex-col`}>
              <div className="mb-1.5 flex items-center gap-1.5">
                <div className="p-1 rounded-md bg-blue-500/12 group-hover/stat:bg-blue-500/20 transition-colors flex-shrink-0">
                  <Calendar className={`${dense ? 'h-3 w-3' : 'h-3.5 w-3.5'} text-blue-600 dark:text-blue-400`} />
                </div>
                <span className="text-[9px] font-bold text-muted-foreground uppercase tracking-wide truncate">Vida útil</span>
              </div>
              <p className={`${dense ? 'text-base' : 'text-lg'} font-bold text-foreground leading-tight truncate`}>
                {equipment.vida_util_meses ? `${equipment.vida_util_meses}m` : "N/A"}
              </p>
              <p className="text-[9px] text-muted-foreground mt-1 leading-snug truncate">
                {equipment.vida_util_meses ? 'Durabilidade' : 'Não informado'}
              </p>
            </div>

            {/* Confiança */}
            <div 
              className={`group/stat rounded-lg border border-border/50 ${confidenceConfig.bg} ${dense ? 'p-2.5' : 'p-3'} hover:shadow-sm transition-all duration-300 hover:-translate-y-0.5 cursor-help flex flex-col`}
              title="Confiança calculada pelo modelo TF-IDF híbrido"
            >
              <div className="mb-1.5 flex items-center gap-1.5">
                <div className={`p-1 rounded-md ${confidenceConfig.bg} group-hover/stat:scale-105 transition-transform flex-shrink-0`}>
                  <TrendingUp className={`${dense ? 'h-3 w-3' : 'h-3.5 w-3.5'} ${confidenceConfig.color}`} />
                </div>
                <span className="text-[9px] font-bold text-muted-foreground uppercase tracking-wide truncate flex-1">Confiança</span>
                <Info className="h-2.5 w-2.5 text-muted-foreground/50 flex-shrink-0" />
              </div>
              <div className="flex items-baseline gap-1">
                <p className={`${dense ? 'text-base' : 'text-lg'} font-bold ${confidenceConfig.color} leading-tight truncate`}>
                  {equipment.confianca ? `${equipment.confianca}%` : "N/A"}
                </p>
                <span className="text-xs flex-shrink-0">{confidenceConfig.icon}</span>
              </div>
              <p className="text-[9px] text-muted-foreground mt-1 leading-snug truncate">
                {confidenceConfig.label}
              </p>
            </div>
          </div>

          {/* Manutenção - altura fixa */}
          <div className={`relative overflow-hidden flex items-center justify-between ${dense ? 'p-2.5' : 'p-3'} rounded-lg border ${maintenanceConfig.bg} hover:shadow-sm transition-all duration-300 group/maintenance min-h-[48px]`}>
            <div className="flex items-center gap-2.5 min-w-0 flex-1">
              <div className={`p-1.5 rounded-md ${maintenanceConfig.bg} group-hover/maintenance:scale-105 transition-transform flex-shrink-0`}>
                <Wrench className="h-3.5 w-3.5" />
              </div>
              <div className="min-w-0 flex-1">
                <span className="text-sm font-bold block leading-none mb-0.5 truncate">Manutenção</span>
                <span className="text-[9px] text-muted-foreground truncate block">{maintenanceConfig.description}</span>
              </div>
            </div>
            <Badge variant="outline" className={`${maintenanceConfig.bg} font-bold ${dense ? 'px-2.5 py-0.5 text-xs' : 'px-3 py-1 text-xs'} shadow-sm flex-shrink-0 ml-2`}>
              {maintenanceConfig.icon} {maintenanceConfig.label}
            </Badge>
          </div>
        </div>
      </CardContent>

      <CardFooter className={`${dense ? 'pt-0 pb-3.5 px-4' : 'pt-0 pb-4 px-5'} flex gap-2 flex-shrink-0`}>
        {/* Botão de adicionar */}
        <Button
          type="button"
          onClick={handleAdd}
          disabled={isAdding}
          className={`flex-1 btn-interactive shadow-md hover:shadow-lg bg-gradient-to-r from-primary via-primary/95 to-primary/90 hover:from-primary/95 hover:to-primary text-primary-foreground font-bold transition-all duration-300 ${
            isAdding ? 'scale-95 opacity-80' : ''
          }`}
        >
          {isAdding ? (
            <>
              <CheckCircle2 className="mr-2 h-4 w-4 animate-pulse-glow" />
              Adicionado!
            </>
          ) : (
            <>
              <ShoppingCart className="mr-2 h-4 w-4" />
              Adicionar
            </>
          )}
        </Button>

        {/* Botão de detalhes */}
        {equipment.link_detalhes && equipment.link_detalhes !== '#' && (
          <>
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => setIsDetailsOpen(true)}
              className="hover:bg-primary/12 hover:border-primary/40 hover:text-primary transition-all duration-300 shadow-md hover:shadow-lg hover:scale-105"
              aria-label="Ver detalhes"
            >
              <Info className="h-4 w-4" />
            </Button>
            <Dialog.Root open={isDetailsOpen} onOpenChange={setIsDetailsOpen}>
              <Dialog.Portal>
                <Dialog.Overlay className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm" />
                <Dialog.Content className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[92vw] max-w-lg rounded-xl border border-border bg-card p-5 shadow-2xl focus:outline-none">
                  <div className="flex items-start justify-between gap-4 mb-4">
                    <Dialog.Title className="text-lg font-semibold leading-tight">{equipment.sugeridos}</Dialog.Title>
                    <button
                      onClick={() => setIsDetailsOpen(false)}
                      className="h-8 w-8 rounded-lg hover:bg-muted/50 flex items-center justify-center transition-colors"
                      aria-label="Fechar"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>

                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-primary/5 to-transparent border border-primary/15">
                      <div className="flex items-center gap-2">
                        <DollarSign className="h-4 w-4 text-primary" />
                        <span className="text-sm font-medium">Valor Unitário</span>
                      </div>
                      <span className="text-sm font-semibold">
                        {formatPrice(equipment.valor_unitario)}
                      </span>
                    </div>
                  
                    <div className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-blue-500/5 to-transparent border border-blue-500/15">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                        <span className="text-sm font-medium">Vida Útil</span>
                      </div>
                      <span className="text-sm font-semibold">{equipment.vida_util_meses ? `${equipment.vida_util_meses} meses` : 'N/A'}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-orange-500/5 to-transparent border border-orange-500/15">
                      <div className="flex items-center gap-2">
                        <Wrench className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                        <span className="text-sm font-medium">Manutenção</span>
                      </div>
                      <span className="text-sm font-semibold">{equipment.manutencao_percent != null ? `${equipment.manutencao_percent}%` : 'N/A'}</span>
                    </div>
                  </div>    
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
          </>
        )}
      </CardFooter>

      {/* Efeito de brilho no hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/0 via-primary/0 to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
    </Card>
  )
}
