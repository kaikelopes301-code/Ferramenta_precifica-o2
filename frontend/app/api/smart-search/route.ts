import { NextRequest, NextResponse } from 'next/server'

type RequestPayload = {
  q?: string
  descricao?: string
  top_k?: number
}

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'
const INITIAL_TIMEOUT = 30000 // 30s para primeira carga de modelos
const NORMAL_TIMEOUT = 15000  // 15s para buscas normais
const MAX_RETRIES = 1

export async function POST(req: NextRequest) {
  const startTime = Date.now()

  try {
    const body = (await req.json().catch(() => ({}))) as RequestPayload
    const descricao = (body.q || body.descricao || '').toString().slice(0, 160)

    const top_k_raw = Number(body.top_k ?? 10)
    const top_k = Math.max(1, Math.min(10, Number.isNaN(top_k_raw) ? 10 : top_k_raw))

    if (!descricao) {
      return NextResponse.json(
        {
          resultados: [],
          tempo_processamento_ms: Date.now() - startTime,
        },
        { status: 200 },
      )
    }

    const backendUrl = `${BACKEND_URL}/buscar-inteligente`

    let lastError: unknown = null
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      const isFirstAttempt = attempt === 0
      const timeout = isFirstAttempt ? INITIAL_TIMEOUT : NORMAL_TIMEOUT

      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), timeout)

      try {
        const response = await fetch(backendUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify({ descricao, top_k }),
          signal: controller.signal,
        })

        clearTimeout(timer)

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          return NextResponse.json(
            {
              resultados: [],
              error: 'backend_error',
              detail: (errorData as { detail?: string }).detail || 'Erro no processamento',
              tempo_processamento_ms: Date.now() - startTime,
            },
            { status: response.status },
          )
        }

        const data = await response.json().catch(() => ({
          resultados: [],
          error: 'invalid_json',
        }))

        const processingTime = Date.now() - startTime
        console.log(`✔️ Busca: ${processingTime}ms - "${descricao.slice(0, 30)}..."`)

        return NextResponse.json(
          {
            ...data,
            tempo_processamento_ms: processingTime,
          },
          {
            status: 200,
            headers: {
              'X-Response-Time': `${processingTime}ms`,
              'Cache-Control': 'no-store',
            },
          },
        )
      } catch (e) {
        clearTimeout(timer)
        lastError = e

        const isTimeout = e instanceof DOMException && e.name === 'AbortError'
        const isNetworkError = e instanceof Error && e.message.includes('fetch')
        if (!isTimeout && !isNetworkError) break
        if (attempt === MAX_RETRIES) break
        await new Promise((resolve) => setTimeout(resolve, 500))
      }
    }

    const processingTime = Date.now() - startTime
    const isTimeout = lastError instanceof DOMException && lastError.name === 'AbortError'
    if (isTimeout || processingTime > 10000) {
      console.warn(`⚠️ Timeout: ${processingTime}ms - "${descricao.slice(0, 30)}..."`)
    }

    return NextResponse.json(
      {
        resultados: [],
        warning: isTimeout ? 'timeout' : 'error',
        detail: 'Backend demorou muito. Tente novamente.',
        tempo_processamento_ms: processingTime,
      },
      {
        status: 200,
        headers: {
          'X-Response-Time': `${processingTime}ms`,
          'X-Fallback': 'true',
        },
      },
    )
  } catch (e) {
    const processingTime = Date.now() - startTime
    const message = e instanceof Error ? e.message : 'unknown_error'
    console.error(`❌ Erro crítico: ${message}`)

    return NextResponse.json(
      {
        resultados: [],
        error: 'critical_error',
        detail: 'Erro interno. Contate o suporte.',
        tempo_processamento_ms: processingTime,
      },
      {
        status: 500,
        headers: {
          'X-Response-Time': `${processingTime}ms`,
        },
      },
    )
  }
}
