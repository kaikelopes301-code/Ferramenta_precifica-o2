import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData()
    const r = await fetch(`${BACKEND_URL}/upload`, {
      method: 'POST',
      body: form,
    })
    const data = await r.json().catch(()=> ({ success: false }))
    return NextResponse.json(data, { status: r.status })
  } catch (e: any) {
    return NextResponse.json({ success: false, error: 'proxy_error', detail: e?.message }, { status: 500 })
  }
}
