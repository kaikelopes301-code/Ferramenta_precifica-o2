#!/usr/bin/env python3
"""
kaike_audit.py — Auditoria universal de projetos (foco em otimização e velocidade)

Como usar (exemplos):
  python kaike_audit.py /caminho/do/projeto
  (python kaike_audit.py . --entry run_local.py --entry-args "--modo=prod")
  python kaike_audit.py . --report out/relatorio.md --profile 10

Saídas:
  - relatório Markdown (kaike_audit_report.md por padrão)
  - JSON estruturado (kaike_audit_report.json)
  - Perfil cProfile opcional (kaike_profile.stats + kaike_profile.txt)

O script tenta usar ferramentas externas se presentes (ruff, radon, bandit, pip-audit),
mas funciona só com biblioteca padrão também. Nenhuma dependência é obrigatória.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# ---------------------------- Utilidades básicas ---------------------------- #

ANSI = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "blue": "\033[94m",
    "reset": "\033[0m",
}

def info(msg: str):
    print(f"{ANSI['blue']}[i]{ANSI['reset']} {msg}")

def warn(msg: str):
    print(f"{ANSI['yellow']}[!]{ANSI['reset']} {msg}")

def err(msg: str):
    print(f"{ANSI['red']}[x]{ANSI['reset']} {msg}")

# ---------------------------- Estruturas de dados --------------------------- #

@dataclass
class Finding:
    category: str
    severity: str  # LOW | MEDIUM | HIGH | CRITICAL
    file: Optional[str]
    line: Optional[int]
    message: str
    hint: Optional[str] = None

@dataclass
class Metrics:
    total_files: int = 0
    python_files: int = 0
    sql_files: int = 0
    js_ts_files: int = 0
    html_css_files: int = 0
    total_loc: int = 0

@dataclass
class Report:
    project: str
    metrics: Metrics
    findings: List[Finding]
    tools: Dict[str, Any]
    profiling: Dict[str, Any]

# ---------------------------- Execução de comandos ------------------------- #

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 120) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"Comando não encontrado: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", "Tempo excedido"

# ---------------------------- Descoberta de projeto ------------------------ #

IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}

PYTHON_PAT = re.compile(r"\.py$")
SQL_PAT = re.compile(r"\.(sql|SQL)$")
JS_TS_PAT = re.compile(r"\.(js|jsx|ts|tsx)$")
HTML_CSS_PAT = re.compile(r"\.(html|css|scss|sass)$")


def walk_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # filtra diretórios ignorados
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for f in filenames:
            yield Path(dirpath) / f

# ---------------------------- Analisadores específicos --------------------- #

# ---- Python AST heurísticas de performance (pandas, loops, etc.) ---- #

PANDAS_SLOW_PATTERNS = [
    (re.compile(r"\.apply\("), "Uso de DataFrame.apply pode ser lento; prefira vetorização ou .map/.merge/.assign."),
    (re.compile(r"\.iterrows\("), "iterrows() é muito lento; use itertuples(index=False) ou vetorização."),
    (re.compile(r"(df|dataframe).*\.append\(", re.I), "DataFrame.append em loop é O(n^2); acumule em lista e use pd.concat uma vez."),
    # read_excel is expected in data ingestion modules
    # Skip to_excel - common legitimate usage for file downloads
]

SQL_ANTIPATTERNS = [
    (re.compile(r"SELECT\s+\*", re.I), "Evite SELECT *; selecione colunas necessárias para reduzir IO."),
    (re.compile(r"WHERE\s+1\s*=\s*1", re.I), "Condicional redundante; pode ocultar filtros ausentes."),
    (re.compile(r"JOIN\s+[^\n]+\s+ON\s+1\s*=\s*1", re.I), "JOIN cartesiano; verifique chaves e índices."),
]

FASTAPI_HINTS = [
    (re.compile(r"from\s+fastapi\s+import"), "Se for produção, preferir gunicorn/uvicorn workers > 1; ative keepalive e HTTP pipelining."),
]

PYODBC_HINTS = [
    (re.compile(r"import\s+pyodbc"), "Para inserts em massa: cursor.fast_executemany = True e parâmetro arraysize."),
]

@dataclass
class PyFileStat:
    path: str
    loc: int
    funcs: int
    classes: int
    avg_func_len: float
    max_func_len: int


def analyze_python_file(path: Path) -> Tuple[PyFileStat, List[Finding]]:
    findings: List[Finding] = []
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return PyFileStat(str(path), 0, 0, 0, 0.0, 0), [Finding("IO", "LOW", str(path), None, f"Falha ao ler arquivo: {e}")]

    loc = src.count("\n") + 1
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        findings.append(Finding("Python", "MEDIUM", str(path), e.lineno, f"Sintaxe inválida: {e.msg}"))
        return PyFileStat(str(path), loc, 0, 0, 0.0, 0), findings

    func_lengths: List[int] = []
    funcs = 0
    classes = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            funcs += 1
            # estima linhas da função
            end = getattr(node, 'end_lineno', node.lineno)
            func_lengths.append(max(0, end - node.lineno + 1))
        elif isinstance(node, ast.AsyncFunctionDef):
            funcs += 1
            end = getattr(node, 'end_lineno', node.lineno)
            func_lengths.append(max(0, end - node.lineno + 1))
        elif isinstance(node, ast.ClassDef):
            classes += 1

    avg_func_len = (sum(func_lengths) / len(func_lengths)) if func_lengths else 0.0
    max_func_len = max(func_lengths) if func_lengths else 0

    # Heurísticas de strings (skip benchmark e test files)
    skip_patterns = any(pattern in str(path).lower() for pattern in ['benchmark', 'test_', '_test', 'tests/', 'scripts/smoke', 'quality_gates', 'kaike_audit'])
    
    if not skip_patterns:
        for pat, hint in PANDAS_SLOW_PATTERNS:
            if pat.search(src):
                findings.append(Finding("Pandas", "HIGH", str(path), None, f"Padrão potencialmente lento detectado", hint))

    for pat, hint in FASTAPI_HINTS:
        if pat.search(src):
            findings.append(Finding("FastAPI", "MEDIUM", str(path), None, "Projeto FastAPI detectado", hint))

    for pat, hint in PYODBC_HINTS:
        if pat.search(src):
            findings.append(Finding("SQLServer", "HIGH", str(path), None, "Conexão pyodbc detectada", hint))

    # Warn sobre funções gigantes
    if max_func_len > 80:
        findings.append(Finding("Complexidade", "MEDIUM", str(path), None, f"Função(s) muito longas (máx {max_func_len} linhas)",
                                "Quebre em unidades menores; melhora legibilidade e hot paths."))

    return PyFileStat(str(path), loc, funcs, classes, avg_func_len, max_func_len), findings

# ---- SQL (estático) ---- #

def analyze_sql_file(path: Path) -> List[Finding]:
    findings: List[Finding] = []
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return [Finding("IO", "LOW", str(path), None, f"Falha ao ler SQL: {e}")]

    for pat, hint in SQL_ANTIPATTERNS:
        for m in pat.finditer(src):
            line = src[: m.start()].count("\n") + 1
            findings.append(Finding("SQL", "HIGH", str(path), line, f"Antipadrão SQL: '{m.group(0)[:30]}...'", hint))

    # Heurística: sem WHERE em SELECT grande
    for m in re.finditer(r"SELECT\s+.+?FROM\s+.+?(?:;|$)", src, re.I | re.S):
        stmt = m.group(0)
        if re.search(r"JOIN", stmt, re.I) and not re.search(r"WHERE", stmt, re.I):
            line = src[: m.start()].count("\n") + 1
            findings.append(Finding("SQL", "MEDIUM", str(path), line, "JOIN sem WHERE detectado",
                                    "Verifique se a intenção é cartesiano; avalie índices e filtro."))

    return findings

# ---- Frontend (React/Next.js) e HTML/CSS ---- #

def quick_front_checks(root: Path) -> List[Finding]:
    """Análise rápida de projetos frontend (React/Next.js)."""
    findings: List[Finding] = []
    
    # Caminhos importantes
    pkg_json = root / "package.json"
    next_cfg = root / "next.config.js"
    app_dir = root / "app"
    pages_dir = root / "pages"
    public_dir = root / "public"
    
    # Delegar análises para funções específicas
    findings.extend(_check_package_json(pkg_json))
    findings.extend(_check_nextjs_config(next_cfg))
    findings.extend(_scan_frontend_files(root, app_dir, pages_dir))
    findings.extend(_check_directory_structure(pages_dir, app_dir))
    findings.extend(_check_public_assets(public_dir, root))
    
    return findings


def _check_package_json(pkg_json: Path) -> List[Finding]:
    """Verifica package.json para dependências e scripts essenciais."""
    findings: List[Finding] = []
    
    if not pkg_json.exists():
        return findings
        
    try:
        data = json.loads(pkg_json.read_text(encoding="utf-8"))
        deps = data.get("dependencies", {})
        devDeps = data.get("devDependencies", {})
        scripts = data.get("scripts", {})
        
        # Dependências essenciais
        if "react" in deps and "next" in deps:
            # TypeScript
            if "@types/react" not in devDeps and "@types/node" not in devDeps:
                findings.append(Finding("TypeScript", "MEDIUM", str(pkg_json), None,
                                       "@types/react e @types/node ausentes",
                                       "Adicione para melhor DX e type safety."))
            
            # ESLint Next
            if "eslint-config-next" not in devDeps:
                findings.append(Finding("ESLint", "LOW", str(pkg_json), None,
                                       "eslint-config-next ausente",
                                       "Adicione para regras específicas do Next.js."))
            
            # Bundle analyzer
            if "@next/bundle-analyzer" not in devDeps:
                findings.append(Finding("Bundle", "LOW", str(pkg_json), None,
                                       "@next/bundle-analyzer ausente",
                                       "Útil para monitorar tamanho de bundles."))
        
        # Scripts essenciais
        if "next" in deps:
            if "build" not in scripts:
                findings.append(Finding("Scripts", "HIGH", str(pkg_json), None,
                                       "Script build ausente",
                                       "Adicione: \"build\": \"next build\""))
            if "start" not in scripts:
                findings.append(Finding("Scripts", "MEDIUM", str(pkg_json), None,
                                       "Script start ausente",
                                       "Adicione: \"start\": \"next start\""))
    except Exception as e:
        findings.append(Finding("JSON", "LOW", str(pkg_json), None, f"Falha ao ler package.json: {e}"))
    
    return findings


def _check_nextjs_config(next_cfg: Path) -> List[Finding]:
    """Verifica configurações do Next.js."""
    findings: List[Finding] = []
    
    if not next_cfg.exists():
        return findings
        
    try:
        cfg_text = next_cfg.read_text(encoding="utf-8")
        if "experimental" in cfg_text and "appDir" in cfg_text:
            findings.append(Finding("Next.js", "LOW", str(next_cfg), None,
                                   "experimental.appDir detectado",
                                   "Remova (estável desde Next 13.4+)."))
        if not re.search(r"swcMinify\s*:\s*true", cfg_text):
            findings.append(Finding("Next.js", "LOW", str(next_cfg), None,
                                   "swcMinify não habilitado",
                                   "Habilite para bundles menores e builds mais rápidos."))
        if not re.search(r"images\s*:\s*\{[\s\S]*domains\s*:", cfg_text):
            findings.append(Finding("Next.js", "MEDIUM", str(next_cfg), None,
                                   "images.domains ausente",
                                   "Defina domains para otimizar next/image com imagens externas."))
    except Exception as e:
        findings.append(Finding("Next.js", "LOW", str(next_cfg), None, f"Falha ao ler next.config.js: {e}"))
    
    return findings


def _scan_frontend_files(root: Path, app_dir: Path, pages_dir: Path) -> List[Finding]:
    """Escaneia arquivos React/Next.js para padrões e antipadrões."""
    findings: List[Finding] = []
    
    # Varredura de código React/Next
    jsx_tsx_files: List[Path] = []
    for fp in walk_files(root):
        if JS_TS_PAT.search(fp.name):
            jsx_tsx_files.append(fp)

    img_tags = 0
    next_image_imports = 0
    dynamic_imports = 0
    for fp in jsx_tsx_files:
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        img_tags += len(re.findall(r"<img\s", txt))
        next_image_imports += len(re.findall(r"from\s+['\"]next/image['\"]", txt))
        dynamic_imports += len(re.findall(r"from\s+['\"]next/dynamic['\"]", txt))

    if img_tags > 0 and next_image_imports == 0 and (app_dir.exists() or pages_dir.exists()):
        findings.append(Finding("Next.js", "HIGH", str(app_dir if app_dir.exists() else pages_dir), None,
                               "Uso de <img> sem next/image",
                               "Troque por `next/image` (lazy, formatos modernos, resize)."))

    if dynamic_imports == 0 and (app_dir.exists() or pages_dir.exists()):
        findings.append(Finding("Next.js", "LOW", str(app_dir if app_dir.exists() else pages_dir), None,
                               "Nenhum uso de next/dynamic detectado",
                               "Avalie dividir componentes pesados com `next/dynamic`."))
    
    return findings


def _check_directory_structure(pages_dir: Path, app_dir: Path) -> List[Finding]:
    """Verifica estrutura de diretórios do projeto Next.js."""
    findings: List[Finding] = []
    
    # Estrutura de diretórios: app vs pages
    if pages_dir.exists() and not app_dir.exists():
        findings.append(Finding("Next.js", "LOW", str(pages_dir), None,
                               "Projeto usa /pages",
                               "Considere migrar para /app (app router) para server components/roteamento moderno."))
    
    return findings


def _check_public_assets(public_dir: Path, root: Path) -> List[Finding]:
    """Verifica assets grandes no diretório /public."""
    findings: List[Finding] = []
    
    if not public_dir.exists():
        return findings
        
    big_assets: List[Tuple[str, int]] = []
    for fp in public_dir.rglob("*"):
        if fp.is_file():
            try:
                size = fp.stat().st_size
            except Exception:
                continue
            if size >= 700 * 1024 and not fp.name.lower().endswith((".webp", ".avif")):
                big_assets.append((str(fp.relative_to(root)), size))
    
    if big_assets:
        sample = ", ".join([f"{p} ({s//1024}KB)" for p, s in big_assets[:5]])
        findings.append(Finding("Assets", "MEDIUM", str(public_dir), None,
                               f"Arquivos grandes no /public: {len(big_assets)} encontrados",
                               f"Converta para WEBP/AVIF e reduza dimensões. Exemplos: {sample}"))
    
    return findings

# ---- Integração com ferramentas externas (opcional) ---- #

def run_tool_if_available(name: str, args: List[str], cwd: Path) -> Tuple[bool, str]:
    code, out, err_ = run_cmd([name] + args, cwd)
    if code == 0:
        return True, out
    elif code in (124, 127):
        return False, ""
    else:
        # Alguns linters retornam código !=0 quando há findings — ainda assim retornamos output
        return True, out or err_

# ---------------------------- Perfil (cProfile) ---------------------------- #

def run_profile(entry: Optional[str], entry_args: Optional[str], seconds: int, cwd: Path) -> Dict[str, Any]:
    if not entry:
        return {"enabled": False, "note": "Nenhum entry point fornecido"}
    try:
        import cProfile
        import pstats
        import shlex
        profile_path = cwd / "kaike_profile.stats"
        txt_path = cwd / "kaike_profile.txt"

        # Monta comando (executa o arquivo como script)
        cmd = [sys.executable, str(cwd / entry)]
        if entry_args:
            cmd += shlex.split(entry_args)

        info(f"Profilando por ~{seconds}s: {' '.join(cmd)}")
        # Executa com tempo limite — se exceder, interrompe sem falhar
        start = time.time()
        pr = cProfile.Profile()
        pr.enable()
        try:
            p = subprocess.Popen(cmd, cwd=cwd)
            while p.poll() is None and (time.time() - start) < seconds:
                time.sleep(0.2)
            if p.poll() is None:
                p.terminate()
        finally:
            pr.disable()
        pr.dump_stats(profile_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            ps = pstats.Stats(profile_path, stream=f).sort_stats("cumulative")
            ps.print_stats(50)
        return {"enabled": True, "profile_file": str(profile_path), "summary_file": str(txt_path)}
    except Exception as e:
        return {"enabled": False, "error": str(e)}

# ---------------------------- Montagem de relatório ------------------------ #

def prioritize(findings: List[Finding]) -> List[Finding]:
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    return sorted(findings, key=lambda f: (order.get(f.severity, 4), f.category, (f.file or ""), (f.line or 0)))

SEVERITY_EMOJI = {"CRITICAL": "🔥", "HIGH": "⚠️", "MEDIUM": "🔧", "LOW": "ℹ️"}


def _build_markdown_header(rep: Report) -> List[str]:
    """Constrói cabeçalho do relatório markdown."""
    fgs = prioritize(rep.findings)
    counts = Counter(f.severity for f in fgs)
    
    return [
        f"# Auditoria de Projeto — {rep.project}\n",
        "## Resumo\n",
        "- Arquivos tot.: {t} (Python: {py}, SQL: {sql}, JS/TS: {js}, HTML/CSS: {hc})".format(
            t=rep.metrics.total_files, py=rep.metrics.python_files, sql=rep.metrics.sql_files,
            js=rep.metrics.js_ts_files, hc=rep.metrics.html_css_files),
        f"- Linhas de código (estimado): {rep.metrics.total_loc}",
        f"- Achados por severidade: CRITICAL={counts.get('CRITICAL',0)}, HIGH={counts.get('HIGH',0)}, MEDIUM={counts.get('MEDIUM',0)}, LOW={counts.get('LOW',0)}\n"
    ]


def _build_tools_section(tools: Dict[str, Any]) -> List[str]:
    """Constrói seção de ferramentas externas."""
    if not tools:
        return []
    
    md = ["## Ferramentas externas (quando disponíveis)\n"]
    for k, v in tools.items():
        md.append(f"### {k}\n")
        content = (v.strip()[:5000]) if isinstance(v, str) else str(v)[:5000]
        md.append("```\n" + content + "\n```")
    
    return md


def _build_findings_section(findings: List[Finding]) -> List[str]:
    """Constrói seção de achados e recomendações."""
    md = ["## Achados e Recomendações\n"]
    
    for f in findings:
        emoji = SEVERITY_EMOJI.get(f.severity, "•")
        loc = f"{Path(f.file).as_posix()}:{f.line}" if f.file and f.line else (f.file or "")
        md.append(f"- {emoji} **[{f.severity}] {f.category}** — {f.message}  ")
        if loc:
            md.append(f"  _Local_: {loc}  ")
        if f.hint:
            md.append(f"  _Dica_: {f.hint}")
    
    md.append("")
    return md


def _build_profiling_section(profiling: Dict[str, Any]) -> List[str]:
    """Constrói seção de profiling."""
    md = ["## Profiling\n"]
    
    if profiling.get("enabled"):
        md.append("Perfil gerado: veja `kaike_profile.txt` para o top 50 por tempo cumulativo.")
    else:
        md.append("Profiling não executado: " + (profiling.get("note") or profiling.get("error", "")))
    
    return md


def to_markdown(rep: Report) -> str:
    """Gera relatório markdown - refatorado de 122→25 linhas com helpers."""
    fgs = prioritize(rep.findings)
    md = []
    
    # Seções delegadas para funções específicas
    md.extend(_build_markdown_header(rep))
    md.extend(_build_tools_section(rep.tools))
    md.extend(_build_findings_section(fgs))
    md.extend(_build_profiling_section(rep.profiling))

    # Playbook de otimização
    md.extend(_build_optimization_playbook())
    
    return "\n".join(md)


def _build_optimization_playbook() -> List[str]:
    """Constrói playbook de otimização."""
    return [
        "## Playbook de Otimização (atalhos práticos)\n",
        textwrap.dedent(
            """
            **Pandas**
            - Troque `apply/iterrows` por operações vetorizadas; prefira `merge`, `map`, `where`, `assign`.
            - Salve intermediários em Parquet (`.to_parquet`) e leia com `pyarrow`; é 10–100x mais rápido que Excel.
            - Use `categorical` em colunas com alta repetição antes de `merge`/`groupby`.
            - Evite `DataFrame.append` em loop; acumule listas e `pd.concat` uma vez.

            **SQL Server (pyodbc)**
            - Para inserts em massa: `cursor.fast_executemany = True` e `cursor.executemany` com listas grandes.
            - Use `READ_COMMITTED_SNAPSHOT` no banco (se possível) para reduzir bloqueios de leitura.
            - Não use `SELECT *`; crie índices cobrindo filtros/joins críticos.

            **FastAPI/Backend**
            - Em produção: múltiplos workers (`gunicorn -k uvicorn.workers.UvicornWorker -w N`), keep-alive e compressão.
            - Reutilize conexões (pooling) e mova IO pesado para tasks assíncronas/filas.

            **Geral**
            - Logue tempos por etapa; foque nos 20% que consomem 80% do tempo.
            - Automatize testes de desempenho (baseline + regressão).
            """
        )
    ]

# ---------------------------- Pipeline principal --------------------------- #

def _scan_files(root: Path) -> Tuple[Metrics, List[Path]]:
    """Escaneia arquivos e coleta métricas básicas."""
    metrics = Metrics()
    all_files = list(walk_files(root))
    metrics.total_files = len(all_files)
    
    for fp in all_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            metrics.total_loc += text.count("\n") + 1
        except Exception:
            pass
        
        if PYTHON_PAT.search(fp.name): 
            metrics.python_files += 1
        elif SQL_PAT.search(fp.name): 
            metrics.sql_files += 1
        elif JS_TS_PAT.search(fp.name): 
            metrics.js_ts_files += 1
        elif HTML_CSS_PAT.search(fp.name): 
            metrics.html_css_files += 1
    
    return metrics, all_files


def _analyze_sources(files: List[Path]) -> Tuple[List[Finding], List[PyFileStat]]:
    """Analisa arquivos de código e coleta findings."""
    findings, py_stats = [], []
    
    for fp in files:
        try:
            size_ok = fp.stat().st_size < 40 * 1024 * 1024  # ignora arquivos gigantes
        except Exception:
            size_ok = True
        
        if not size_ok:
            findings.append(Finding("Repo", "LOW", str(fp), None, 
                                    "Arquivo muito grande na árvore do projeto", 
                                    "Avalie uso de LFS/artefatos externos."))
            continue

        if PYTHON_PAT.search(fp.name):
            stat, fnds = analyze_python_file(fp)
            py_stats.append(stat)
            findings.extend(fnds)
        elif SQL_PAT.search(fp.name):
            findings.extend(analyze_sql_file(fp))
    
    return findings, py_stats


def _postprocess_findings(findings: List[Finding], py_stats: List[PyFileStat], root: Path) -> List[Finding]:
    """Processa findings finais e adiciona checks de complexidade."""
    # Sumarização de estatísticas Python
    if py_stats:
        worst = max(py_stats, key=lambda s: s.max_func_len)
        if worst.max_func_len > 120:
            findings.append(Finding("Complexidade", "HIGH", worst.path, None,
                                    f"Função extrema detectada (até {worst.max_func_len} linhas)",
                                    "Refatore em funções menores e puras, favoreça composição."))
    
    # Checks de frontend
    findings.extend(quick_front_checks(root))
    return findings


def _run_external_tools(root: Path) -> Dict[str, Any]:
    """Executa ferramentas externas de análise."""
    tools_out: Dict[str, Any] = {}
    
    ok, out = run_tool_if_available("ruff", ["check", "--quiet", str(root)], root)
    if ok and out.strip(): 
        tools_out["ruff"] = out
    
    ok, out = run_tool_if_available("radon", ["cc", "-s", "-a", str(root)], root)
    if ok and out.strip(): 
        tools_out["radon (complexidade)"] = out
    
    ok, out = run_tool_if_available("bandit", ["-r", str(root), "-q"], root)
    if ok and out.strip(): 
        tools_out["bandit (segurança)"] = out
    
    ok, out = run_tool_if_available("pip-audit", [], root)
    if ok and out.strip(): 
        tools_out["pip-audit (vulns)"] = out
    
    return tools_out


def audit_project(root: Path, entry: Optional[str], entry_args: Optional[str], profile_seconds: int) -> Report:
    """Função principal de auditoria - refatorada de 122→35 linhas com helpers."""
    info("Varredura de arquivos...")
    metrics, files = _scan_files(root)
    
    findings, py_stats = _analyze_sources(files)
    findings = _postprocess_findings(findings, py_stats, root)
    
    info("Rodando ferramentas externas se disponíveis (ruff, radon, bandit, pip-audit)...")
    tools_out = _run_external_tools(root)
    
    # Profiling opcional
    if profile_seconds > 0:
        profiling = run_profile(entry, entry_args, profile_seconds, root)
    else:
        profiling = {"enabled": False, "note": "Desligado"}

    return Report(project=str(root.resolve()), metrics=metrics, findings=findings, tools=tools_out, profiling=profiling)

# ---------------------------- CLI ----------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Auditoria universal de projetos (kaike)")
    ap.add_argument("project", help="Diretório raiz do projeto")
    ap.add_argument("--entry", help="Arquivo de entrada para profiling (ex.: run_local.py)")
    ap.add_argument("--entry-args", help="Argumentos para o entry point", default=None)
    ap.add_argument("--report", help="Caminho do relatório Markdown", default="kaike_audit_report.md")
    ap.add_argument("--json", help="Caminho do JSON de saída", default="kaike_audit_report.json")
    ap.add_argument("--profile", type=int, default=0, help="Segundos de profiling (0 para desativar)")

    args = ap.parse_args()
    root = Path(args.project).resolve()
    if not root.exists() or not root.is_dir():
        err("Diretório do projeto inválido")
        sys.exit(2)

    report = audit_project(root, args.entry, args.entry_args, args.profile)

    # Salvar Markdown e JSON
    md = to_markdown(report)
    Path(args.report).write_text(md, encoding="utf-8")
    Path(args.json).write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")

    info(f"Relatório salvo em: {args.report}")
    info(f"JSON salvo em: {args.json}")
    if report.profiling.get("enabled"):
        info("Arquivos de perfil: kaike_profile.stats + kaike_profile.txt")

if __name__ == "__main__":
    main()
