"use client"

import * as Dialog from "@radix-ui/react-dialog"
import { DollarSign, Calendar, Wrench, X } from "lucide-react"
import type { Equipment } from "@/app/page"

interface EquipmentCardDialogProps {
  open: boolean
  onOpenChange: (next: boolean) => void
  equipment: Equipment
  formatPrice: (price: number | null) => string
}

export function EquipmentCardDialog({ open, onOpenChange, equipment, formatPrice }: EquipmentCardDialogProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm" />
        <Dialog.Content className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[92vw] max-w-lg rounded-xl border border-border bg-card p-5 shadow-2xl focus:outline-none">
          <div className="flex items-start justify-between gap-4 mb-4">
            <Dialog.Title className="text-lg font-semibold leading-tight">{equipment.sugeridos}</Dialog.Title>
            <button
              onClick={() => onOpenChange(false)}
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
                <span className="text-sm font-medium">Vida útil</span>
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
  )
}
