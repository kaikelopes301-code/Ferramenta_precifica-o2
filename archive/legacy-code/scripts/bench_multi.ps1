$ErrorActionPreference='Stop'
Set-Location "$PSScriptRoot\.."
if (Test-Path .venv\Scripts\Activate.ps1) { . .venv\Scripts\Activate.ps1 }
$srv = Start-Process -PassThru -WindowStyle Hidden python -ArgumentList '-m','uvicorn','src.api.main:app','--host','127.0.0.1','--port','8001'
# Espera /health
$ready = $false
for($i=0; $i -lt 40; $i++){
  try { $h = Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8001/health' -TimeoutSec 2; if($h -and $h.status -eq 'ok'){ $ready=$true; break } } catch {}
  Start-Sleep -Milliseconds 500
}
if(-not $ready){ Write-Output 'server_start_failed=true'; Stop-Process -Id $srv.Id -Force; exit 1 }

$payload = '{"descricao":"mop industrial","top_k":8}'
$results = @()
for($i=1; $i -le 5; $i++){
  $sw=[System.Diagnostics.Stopwatch]::StartNew()
  try{ $r = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/buscar-inteligente' -ContentType 'application/json' -Body $payload -TimeoutSec 60 } catch { $r=$null }
  $sw.Stop()
  $elapsed=[math]::Round($sw.Elapsed.TotalMilliseconds)
  if($r -and $r.resultados){ $sug=$r.resultados[0].sugeridos; $sc=$r.resultados[0].score } else { $sug='N/A'; $sc=$null }
  Write-Output ("run=$i elapsed_ms=$elapsed top1=$sug score=$sc")
  $results += $elapsed
}
# Estatísticas simples
$min = ($results | Measure-Object -Minimum).Minimum
$max = ($results | Measure-Object -Maximum).Maximum
$avg = [math]::Round((($results | Measure-Object -Average).Average),0)
Write-Output ("summary min_ms=$min max_ms=$max avg_ms=$avg")
Stop-Process -Id $srv.Id -Force
