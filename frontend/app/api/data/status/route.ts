import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function GET() {
  try {
    const r = await fetch(`${BACKEND_URL}/data/status`, { cache: 'no-store' })
    const data = await r.json().catch(()=> ({ has_data: false }))
    return NextResponse.json(data, { status: r.status })
  } catch (e: any) {
    return NextResponse.json({ has_data: false, error: 'proxy_error', detail: e?.message }, { status: 200 })
  }
}
