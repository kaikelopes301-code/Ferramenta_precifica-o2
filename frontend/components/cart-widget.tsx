"use client"

import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Download, ShoppingCart, Trash2, X, Plus, Minus, Pencil, Check, Copy, FileSpreadsheet } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export type CartItem = {
  id: string
  name: string
  price: number | null
  qty: number
  notes?: string
  vidaUtilMeses?: number | null
  manutencaoPercent?: number | null
  fornecedor?: string | null
  marca?: string | null
  descricao?: string | null
}

const currencyBRL = new Intl.NumberFormat("pt-BR", {
  style: "currency",
  currency: "BRL",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

interface CartWidgetProps {
  items: CartItem[]
  onClear: () => void
  onRemove: (id: string) => void
  onChangeQty: (id: string, qty: number) => void
  onChangeNotes: (id: string, notes: string) => void
  onChangeName: (id: string, name: string) => void
}

export function CartWidget({ items, onClear, onRemove, onChangeQty, onChangeNotes, onChangeName }: CartWidgetProps) {
  const [open, setOpen] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingValue, setEditingValue] = useState<string>("")
  const { toast } = useToast()

  const totals = useMemo(() => {
    const totalQty = items.reduce((acc, it) => acc + (it.qty || 0), 0)
    const totalPrice = items.reduce((acc, it) => acc + ((it.price || 0) * (it.qty || 0)), 0)
    return { totalQty, totalPrice }
  }, [items])

  // Removido: não abrir automaticamente o carrinho ao adicionar um item
  // Caso futuramente queira reativar via evento, adicione um listener aqui.

  const copyToClipboard = async () => {
    try {
      const resumoMap = new Map<
        string,
        { desc: string | null; name: string; price: number | null; qty: number; vida: number | null; manperc: number | null }
      >()

      for (const it of items) {
        const key = `${it.descricao || ""}__${it.id}`
        const ex = resumoMap.get(key)
        if (ex) {
          resumoMap.set(key, { ...ex, qty: ex.qty + (it.qty || 0) })
        } else {
          resumoMap.set(key, {
            desc: it.descricao || "",
            name: it.name,
            price: it.price ?? null,
            qty: it.qty || 0,
            vida: it.vidaUtilMeses ?? null,
            manperc: it.manutencaoPercent ?? null,
          })
        }
      }

      let valorTotal = 0
      resumoMap.forEach((v) => {
        const sub = v.price != null ? v.price * v.qty : 0
        valorTotal += sub
      })

      let text = "RELATÓRIO DE PRECIFICAÇÃO - CARRINHO\n"
      text += `Gerado em: ${new Date().toLocaleString("pt-BR")}\n`
      text += "─".repeat(80) + "\n\n"

      text += `RESUMO\n`
      text += `Itens únicos:\t${resumoMap.size}\n`
      text += `Quantidade total:\t${items.reduce((acc, it) => acc + (it.qty || 0), 0)}\n`
      text += `Valor total:\t${currencyBRL.format(valorTotal)}\n\n`
      text += "─".repeat(80) + "\n\n"

      text += "ITENS DETALHADOS\n\n"
      text += "Descrição (Usuário)\tSugestão\tQtd\tVida útil (meses)\tPreço Unit.\tManutenção (%)\tSubtotal\n"
      text += "─".repeat(80) + "\n"

      resumoMap.forEach((v) => {
        const sub = v.price != null ? v.price * v.qty : null
        text += `${v.desc || ""}\t${v.name}\t${v.qty}\t${v.vida ?? ""}\t${
          v.price != null ? currencyBRL.format(v.price) : ""
        }\t${v.manperc != null ? v.manperc.toFixed(1) + "%" : ""}\t${
          sub != null ? currencyBRL.format(sub) : ""
        }\n`
      })

      text += "─".repeat(80) + "\n"
      text += `TOTAL\t\t\t\t\t\t${currencyBRL.format(valorTotal)}\n`

      await navigator.clipboard.writeText(text)
      toast({
        title: "✔ Copiado!",
        description: "Tabela copiada. Cole em Excel, Word ou outro aplicativo.",
      })
    } catch (error) {
      console.error("Erro ao copiar:", error)
      toast({
        title: "Erro ao copiar",
        description: "Não foi possível copiar para a área de transferência.",
        variant: "destructive",
      })
    }
  }

  const exportExcel = async () => {
    try {
      toast({ title: "📄 Gerando arquivo", description: "Preparando planilhas…" })

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
      a.download = "Resultados.xlsx"
      a.click()
      URL.revokeObjectURL(url)

      toast({ title: "✔ Exportado com sucesso", description: "Arquivo Resultados.xlsx gerado." })
      return
    } catch (e) {
      // Fallback para CSV caso o Excel falhe (ou backend esteja indisponível)
      try {
        const headers = [
          "Descrição (Usuário)",
          "Sugestão",
          "Quantidade",
          "Vida útil (meses)",
          "Preço Unitário",
          "Manutenção (%)",
          "Subtotal",
        ]
        const resumoMap = new Map<
          string,
          { desc: string | null; name: string; price: number | null; qty: number; vida: number | null; manperc: number | null }
        >()
        for (const it of items) {
          const key = `${it.descricao || ""}__${it.id}`
          const ex = resumoMap.get(key)
          if (ex) {
            resumoMap.set(key, { ...ex, qty: ex.qty + (it.qty || 0) })
          } else {
            resumoMap.set(key, {
              desc: it.descricao || "",
              name: it.name,
              price: it.price ?? null,
              qty: it.qty || 0,
              vida: it.vidaUtilMeses ?? null,
              manperc: it.manutencaoPercent ?? null,
            })
          }
        }
        const rows: string[] = []
        resumoMap.forEach((v) => {
          const sub = v.price != null ? v.price * v.qty : null
          rows.push(
            [
              `"${v.desc || ""}"`,
              `"${v.name}"`,
              String(v.qty),
              v.vida ?? "",
              v.price != null ? v.price.toFixed(2) : "",
              v.manperc != null ? v.manperc.toFixed(2) : "",
              sub != null ? sub.toFixed(2) : "",
            ].join(","),
          )
        })
        const csv = [headers.join(","), ...rows].join("\n")
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" })
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = "carrinho.csv"
        a.click()
        URL.revokeObjectURL(url)
        toast({
          title: "✔ Exportado em CSV",
          description: "Não foi possível gerar Excel; usamos CSV como alternativa.",
        })
        return
      } catch (err) {
        console.error("Export error", err)
        toast({
          title: "Falha ao exportar",
          description: "Tente novamente ou atualize a página.",
          variant: "destructive",
        })
      }
    }
  }

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-30 transition-opacity"
          onClick={() => setOpen(false)}
        />
      )}

      <div className="fixed bottom-8 right-8 z-40">
        <button
          onClick={() => setOpen(!open)}
          className="relative bg-white dark:bg-gray-900 text-gray-900 dark:text-white shadow-lg hover:shadow-xl border border-gray-200 dark:border-gray-700 rounded-full h-14 px-6 flex items-center gap-3 transition-all duration-200 hover:scale-105"
        >
          <ShoppingCart className="h-5 w-5" strokeWidth={1.5} />
          <span className="font-medium text-sm">{totals.totalQty}</span>

          {totals.totalQty > 0 && (
            <span className="absolute -top-1 -right-1 bg-black dark:bg-white text-white dark:text-black text-[10px] font-bold rounded-full h-5 w-5 flex items-center justify-center">
              {totals.totalQty}
            </span>
          )}
        </button>

        {open && (
          <div className="absolute bottom-20 right-0 w-[440px] bg-white dark:bg-gray-900 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-800 overflow-hidden">
            <div className="px-6 py-5 border-b border-gray-100 dark:border-gray-800">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Carrinho</h2>
                <button
                  onClick={() => setOpen(false)}
                  className="h-8 w-8 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 flex items-center justify-center transition-colors"
                >
                  <X className="h-4 w-4" strokeWidth={2} />
                </button>
              </div>

              <div className="flex items-baseline justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {items.length} {items.length === 1 ? "item" : "itens"}
                </span>
                <div className="text-right">
                  <div className="text-2xl font-semibold text-gray-900 dark:text-white tracking-tight">
                    {currencyBRL.format(totals.totalPrice)}
                  </div>
                </div>
              </div>
            </div>

            <div className="overflow-y-auto max-h-[50vh] px-6 py-4">
              {items.length === 0 ? (
                <div className="text-center py-16">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-50 dark:bg-gray-800 flex items-center justify-center">
                    <ShoppingCart className="h-7 w-7 text-gray-300 dark:text-gray-600" strokeWidth={1.5} />
                  </div>
                  <p className="text-gray-900 dark:text-white font-medium mb-1">Carrinho vazio</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Adicione itens para começar</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {items.map((it) => (
                    <div
                      key={it.id}
                      className="group pb-4 border-b border-gray-100 dark:border-gray-800 last:border-0 last:pb-0"
                    >
                      <div className="flex gap-4 mb-3">
                        <div className="flex-1 min-w-0">
                          {editingId === it.id ? (
                            <div className="flex items-center gap-2 mb-1">
                              <input
                                value={editingValue}
                                onChange={(e) => setEditingValue(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") {
                                    const newVal = editingValue.trim()
                                    if (newVal) onChangeName(it.id, newVal)
                                    setEditingId(null)
                                  } else if (e.key === "Escape") {
                                    setEditingId(null)
                                  }
                                }}
                                onBlur={() => {
                                  const newVal = editingValue.trim()
                                  if (newVal) onChangeName(it.id, newVal)
                                  setEditingId(null)
                                }}
                                autoFocus
                                className="flex-1 min-w-0 text-sm px-2 py-1 rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-gray-300 dark:focus:ring-gray-600"
                              />
                              <button
                                onClick={() => {
                                  const newVal = editingValue.trim()
                                  if (newVal) onChangeName(it.id, newVal)
                                  setEditingId(null)
                                }}
                                className="h-7 w-7 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 flex items-center justify-center"
                                title="Salvar"
                              >
                                <Check className="h-4 w-4" />
                              </button>
                              <button
                                onClick={() => setEditingId(null)}
                                className="h-7 w-7 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 flex items-center justify-center"
                                title="Cancelar"
                              >
                                <X className="h-4 w-4" />
                              </button>
                            </div>
                          ) : (
                            <div className="flex items-center gap-2 mb-1">
                              <h3 className="font-medium text-gray-900 dark:text-white text-sm leading-tight truncate">
                                {it.name}
                              </h3>
                              <button
                                onClick={() => {
                                  setEditingId(it.id)
                                  setEditingValue(it.name)
                                }}
                                className="opacity-60 hover:opacity-100 h-6 w-6 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 flex items-center justify-center flex-shrink-0"
                                title="Editar nome"
                              >
                                <Pencil className="h-3.5 w-3.5" />
                              </button>
                            </div>
                          )}
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {it.price != null ? currencyBRL.format(it.price) : "—"}
                          </p>
                        </div>
                        <button
                          onClick={() => onRemove(it.id)}
                          className="opacity-0 group-hover:opacity-100 h-7 w-7 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 flex items-center justify-center transition-all flex-shrink-0"
                          title="Remover"
                        >
                          <Trash2 className="h-4 w-4 text-gray-400" strokeWidth={1.5} />
                        </button>
                      </div>

                      <div className="flex items-center justify-between gap-4">
                        <div className="flex items-center gap-2 bg-gray-50 dark:bg-gray-800 rounded-lg p-1">
                          <button
                            onClick={() => onChangeQty(it.id, Math.max(1, it.qty - 1))}
                            className="h-7 w-7 rounded hover:bg-white dark:hover:bg-gray-700 flex items-center justify-center transition-colors"
                            disabled={it.qty <= 1}
                          >
                            <Minus className="h-3 w-3" strokeWidth={2} />
                          </button>
                          <span className="w-8 text-center text-sm font-medium">{it.qty}</span>
                          <button
                            onClick={() => onChangeQty(it.id, it.qty + 1)}
                            className="h-7 w-7 rounded hover:bg-white dark:hover:bg-gray-700 flex items-center justify-center transition-colors"
                          >
                            <Plus className="h-3 w-3" strokeWidth={2} />
                          </button>
                        </div>

                        <div className="text-sm font-semibold text-gray-900 dark:text-white">
                          {it.price != null ? currencyBRL.format(it.price * it.qty) : "—"}
                        </div>
                      </div>

                      <textarea
                        placeholder="Observações"
                        value={it.notes || ""}
                        rows={1}
                        onChange={(e) => onChangeNotes(it.id, e.target.value)}
                        onInput={(e) => {
                          const el = e.currentTarget
                          el.style.height = "auto"
                          const lineH = parseFloat(getComputedStyle(el).lineHeight || "20") || 20
                          const maxH = lineH * 3
                          const nextH = Math.min(el.scrollHeight, maxH)
                          el.style.height = `${nextH}px`
                          el.style.overflowY = el.scrollHeight > maxH ? "auto" : "hidden"
                        }}
                        className="mt-3 w-full text-xs px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 placeholder:text-gray-400 dark:placeholder:text-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-300 dark:focus:ring-gray-600 transition-all resize-none"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {items.length > 0 && (
              <div className="px-6 py-4 border-t border-gray-100 dark:border-gray-800 space-y-2.5">
                <Button
                  onClick={exportExcel}
                  className="w-full h-11 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-xl font-semibold transition-all shadow-md hover:shadow-lg"
                >
                  <FileSpreadsheet className="mr-2 h-4 w-4" strokeWidth={2} />
                  Gerar Planilha Formatada
                </Button>

                <Button
                  onClick={copyToClipboard}
                  variant="outline"
                  className="w-full h-11 border-2 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-xl font-medium transition-all"
                >
                  <Copy className="mr-2 h-4 w-4" strokeWidth={2} />
                  Copiar Tabela
                </Button>

                <button
                  onClick={onClear}
                  className="w-full text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 py-2 transition-colors"
                >
                  Limpar carrinho
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  )
}
