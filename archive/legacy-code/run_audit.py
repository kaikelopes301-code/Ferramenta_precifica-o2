#!/usr/bin/env python3
"""
Script para executar a auditoria do projeto de forma simplificada.
Uso: python run_audit.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    audit_script = project_root / "kaike_audit.py"
    
    if not audit_script.exists():
        print("❌ Arquivo kaike_audit.py não encontrado!")
        return 1
    
    print("🔍 Executando auditoria completa do projeto...")
    print("📁 Projeto:", project_root)
    
    # Executa auditoria sem profiling para evitar problemas no Windows
    cmd = [
        sys.executable, 
        str(audit_script),
        ".",
        "--entry", "src/api/main.py",
        "--report", "audit_report.md",
        "--json", "audit_report.json"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("\n✅ Auditoria concluída com sucesso!")
        print("📄 Relatórios gerados:")
        print("  - audit_report.md (relatório em Markdown)")
        print("  - audit_report.json (dados estruturados)")
        
        # Mostra um resumo rápido
        try:
            import json
            with open(project_root / "audit_report.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            metrics = data["metrics"]
            findings = data["findings"]
            severities = {}
            for f in findings:
                severities[f["severity"]] = severities.get(f["severity"], 0) + 1
            
            print(f"\n📊 Resumo:")
            print(f"  - Arquivos: {metrics['total_files']} total")
            print(f"  - Python: {metrics['python_files']}, JS/TS: {metrics['js_ts_files']}")
            print(f"  - Achados: {len(findings)} total")
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                if sev in severities:
                    emoji = {"CRITICAL": "🔥", "HIGH": "⚠️", "MEDIUM": "🔧", "LOW": "ℹ️"}[sev]
                    print(f"    {emoji} {sev}: {severities[sev]}")
                    
        except Exception as e:
            print(f"⚠️ Erro ao ler resumo: {e}")
            
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro durante auditoria: {e}")
        return e.returncode
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())