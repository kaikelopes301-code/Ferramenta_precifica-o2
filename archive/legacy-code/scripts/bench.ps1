$ErrorActionPreference='Stop'
Set-Location "$PSScriptRoot\.."
if (Test-Path .venv\Scripts\Activate.ps1) { . .venv\Scripts\Activate.ps1 }
$srv = Start-Process -PassThru -WindowStyle Hidden python -ArgumentList '-m','uvicorn','src.api.main:app','--host','127.0.0.1','--port','8001'
# Espera o servidor responder no /health (até ~15s)
$ready = $false
for($i=0; $i -lt 30; $i++){
  try {
    $h = Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8001/health' -TimeoutSec 2
    if($h -and $h.status -eq 'ok'){ $ready = $true; break }
  } catch {}
  Start-Sleep -Milliseconds 500
}
if(-not $ready){ Write-Output 'server_start_failed=true'; Stop-Process -Id $srv.Id -Force; exit 1 }
try {
  $sw=[System.Diagnostics.Stopwatch]::StartNew()
  $body = '{"descricao":"mop industrial","top_k":8}'
  $r = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/buscar-inteligente' -ContentType 'application/json' -Body $body
  $sw.Stop()
  Write-Output ('elapsed_ms='+[math]::Round($sw.Elapsed.TotalMilliseconds))
  if($r -and $r.resultados){
    $first = $r.resultados[0]
    Write-Output ('top1='+$first.sugeridos+'; score='+$first.score)
  } else {
    Write-Output 'no_results=true'
  }
} finally {
  Stop-Process -Id $srv.Id -Force
}
