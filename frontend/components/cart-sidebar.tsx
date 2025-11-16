"use client"

import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Trash2, Download } from "lucide-react"

type CartItem = {
  id: string
  name: string
  price: number | null
  qty: number
  notes?: string
}

interface CartSidebarProps {
  items: CartItem[]
  onChangeQty: (id: string, qty: number) => void
  onChangeNotes: (id: string, notes: string) => void
  onRemove: (id: string) => void
  onClear: () => void
}

const currencyBRL = new Intl.NumberFormat("pt-BR", {
  style: "currency",
  currency: "BRL",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

export function CartSidebar({ items, onChangeQty, onChangeNotes, onRemove, onClear }: CartSidebarProps) {
  const [open, setOpen] = useState(true)

  const totals = useMemo(() => {
    const totalQty = items.reduce((acc, it) => acc + (it.qty || 0), 0)
    const totalPrice = items.reduce((acc, it) => acc + ((it.price || 0) * (it.qty || 0)), 0)
    return { totalQty, totalPrice }
  }, [items])

  const saveDraft = () => {
    try {
      localStorage.setItem("cartDraft", JSON.stringify(items))
    } catch {}
  }

  const exportExcel = async () => {
    try {
      const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000") + "/cart/export"

      const response = await fetch(backendUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ items }),
      })

      if (!response.ok) {
        throw new Error("Erro ao gerar arquivo Excel no backend")
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "carrinho.xlsx"
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error("Falhou exportação de Excel; usando CSV fallback", error)
      const headers = ["Item", "Quantidade", "Preço Unitário", "Subtotal", "Observações"]
      const lines = items.map((it) =>
        [
          `"${it.name.replace(/"/g, '""')}"`,
          it.qty,
          it.price != null ? currencyBRL.format(it.price) : "—",
          it.price != null ? currencyBRL.format(it.price * it.qty) : "—",
          `"${(it.notes || "").replace(/"/g, '""')}"`,
        ].join(","),
      )
      const csv = [headers.join(","), ...lines].join("\n")
      const blob = new Blob([csv], { type: "text/csv" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "carrinho.csv"
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <aside
      className={`fixed right-0 top-0 h-full w-full sm:max-w-md border-l border-border bg-background shadow-xl z-50 transition-transform ${
        open ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="font-semibold">
          Carrinho ({items.length} itens • {totals.totalQty} un)
        </div>
        <button
          className="text-sm text-muted-foreground hover:text-foreground"
          onClick={() => setOpen(!open)}
        >
          {open ? "Fechar" : "Abrir"}
        </button>
      </div>
      <div className="p-4 space-y-3 overflow-y-auto h-[calc(100%-160px)]">
        {items.length === 0 && (
          <p className="text-sm text-muted-foreground">Seu carrinho está vazio.</p>
        )}
        {items.map((it) => (
          <div key={it.id} className="rounded-md border border-border p-3 space-y-2">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="font-medium truncate" title={it.name}>
                  {it.name}
                </div>
                <div className="text-xs text-muted-foreground">
                  Preço: {it.price != null ? currencyBRL.format(it.price) : "—"}
                </div>
              </div>
              <button
                className="p-2 rounded-md hover:bg-destructive/10 text-destructive"
                onClick={() => onRemove(it.id)}
                title="Remover"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground">Qtd</label>
              <Input
                type="number"
                value={it.qty}
                min={1}
                onChange={(e) => onChangeQty(it.id, Math.max(1, parseInt(e.target.value || "1")))}
                className="w-20"
              />
              <div className="ml-auto text-sm font-medium">
                Subtotal: {it.price != null ? currencyBRL.format(it.price * it.qty) : "—"}
              </div>
            </div>
            <div>
              <Input
                placeholder="Observações (opcional)"
                value={it.notes || ""}
                onChange={(e) => onChangeNotes(it.id, e.target.value)}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="p-4 border-t border-border space-y-3">
        <div className="flex items-center justify-between font-semibold">
          <span>Total</span>
          <span>{currencyBRL.format(totals.totalPrice)}</span>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="flex-1" onClick={onClear}>
            Limpar carrinho
          </Button>
          <Button variant="outline" className="flex-1" onClick={saveDraft}>
            Salvar rascunho
          </Button>
        </div>
        <Button className="w-full" onClick={exportExcel}>
          <Download className="mr-2 h-4 w-4" /> Exportar carrinho
        </Button>
      </div>
    </aside>
  )
}

export type { CartItem }
