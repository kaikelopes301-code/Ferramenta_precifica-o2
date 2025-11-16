import { NextRequest, NextResponse } from 'next/server'

/**
 * �️ Smart Search API Route - Versão Robusta com Blindagem
 * • Timeout adaptativo (30s) para carregamento de modelos
 * • Retry automático em caso de timeout
 * • Fallback gracioso com resultados vazios
 * • Logs limpos sem poluição
 */

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'
const INITIAL_TIMEOUT = 30000 // 30s para primeira carga de modelos
const NORMAL_TIMEOUT = 15000  // 15s para buscas normais
const MAX_RETRIES = 1

export async function POST(req: NextRequest) {
  const startTime = Date.now()
  
  try {
    const body = await req.json().catch(() => ({})) as any
    const descricao = (body.q || body.descricao || '').toString().slice(0, 160)
    
    // Protege o backend: limita top_k e normaliza
    const top_k_raw = Number(body.top_k ?? 10)
    const top_k = Math.max(1, Math.min(10, isNaN(top_k_raw) ? 10 : top_k_raw))
    
    if (!descricao) {
      return NextResponse.json({ 
        resultados: [],
        tempo_processamento_ms: Date.now() - startTime
      }, { status: 200 })
    }

    const backendUrl = `${BACKEND_URL}/buscar-inteligente`

    // 🔄 Retry logic com timeout adaptativo
    let lastError: any = null
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      const isFirstAttempt = attempt === 0
      const timeout = isFirstAttempt ? INITIAL_TIMEOUT : NORMAL_TIMEOUT
      
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), timeout)
      
      try {
        const response = await fetch(backendUrl, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify({ descricao, top_k }),
          signal: ctrl.signal,
          // @ts-ignore - keepalive para conexões persistentes
          keepalive: true,
        })
        
        clearTimeout(timer)
        
        if (!response.ok) {
          // Backend retornou erro, mas respondeu
          const errorData = await response.json().catch(() => ({}))
          return NextResponse.json({ 
            resultados: [],
            error: 'backend_error',
            detail: errorData.detail || 'Erro no processamento',
            tempo_processamento_ms: Date.now() - startTime,
          }, { status: response.status })
        }
        
        const data = await response.json().catch(() => ({ 
          resultados: [],
          error: 'invalid_json' 
        }))
        
        // ✅ Sucesso - log limpo apenas
        const processingTime = Date.now() - startTime
        console.log(`✅ Busca: ${processingTime}ms - "${descricao.slice(0, 30)}..."`)
        
        return NextResponse.json({
          ...data,
          tempo_processamento_ms: processingTime
        }, { 
          status: 200,
          headers: {
            'X-Response-Time': `${processingTime}ms`,
            'Cache-Control': 'no-store',
          }
        })
        
      } catch (e: any) {
        clearTimeout(timer)
        lastError = e
        
        const isTimeout = e?.name === 'AbortError'
        const isNetworkError = e?.message?.includes('fetch')
        
        // Se não for timeout ou erro de rede, não retry
        if (!isTimeout && !isNetworkError) break
        
        // Se for última tentativa, quebra o loop
        if (attempt === MAX_RETRIES) break
        
        // Aguarda 500ms antes de retry
        await new Promise(resolve => setTimeout(resolve, 500))
      }
    }
    
    // 🛡️ Fallback gracioso - retorna vazio ao invés de erro
    const processingTime = Date.now() - startTime
    const isTimeout = lastError?.name === 'AbortError'
    
    // Log apenas quando relevante (timeout ou erro crítico)
    if (isTimeout || processingTime > 10000) {
      console.warn(`⚠️ Timeout: ${processingTime}ms - "${descricao.slice(0, 30)}..."`)
    }
    
    return NextResponse.json({ 
      resultados: [],
      warning: isTimeout ? 'timeout' : 'error',
      detail: 'Backend demorou muito. Tente novamente.',
      tempo_processamento_ms: processingTime,
    }, { 
      status: 200, // Retorna 200 para evitar erro no frontend
      headers: {
        'X-Response-Time': `${processingTime}ms`,
        'X-Fallback': 'true',
      }
    })
    
  } catch (e: any) {
    // Erro crítico no parsing ou lógica
    const processingTime = Date.now() - startTime
    console.error(`❌ Erro crítico: ${e?.message}`)
    
    return NextResponse.json({ 
      resultados: [],
      error: 'critical_error',
      detail: 'Erro interno. Contate o suporte.',
      tempo_processamento_ms: processingTime,
    }, { 
      status: 500,
      headers: {
        'X-Response-Time': `${processingTime}ms`,
      }
    })
  }
}
