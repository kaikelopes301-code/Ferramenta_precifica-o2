import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

interface BatchBody {
  descricoes?: string[]
  top_k?: number
  use_tfidf?: boolean
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json().catch(() => ({}))) as BatchBody
    const descricoes = Array.isArray(body.descricoes) ? body.descricoes.slice(0, 200) : []
    const top_k = Math.max(1, Math.min(50, Number(body.top_k ?? 5)))
    const use_tfidf = Boolean(body.use_tfidf)
    const r = await fetch(`${BACKEND_URL}/buscar-lote-inteligente`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-user-id': req.headers.get('x-user-id') || '' },
      body: JSON.stringify({ descricoes, top_k, use_tfidf })
    })
    const data = await r.json().catch(()=> ({ resultados: [] }))
    return NextResponse.json(data, { status: r.status })
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'unknown_error'
    return NextResponse.json({ error: 'proxy_error', detail }, { status: 500 })
  }
}
