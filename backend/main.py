#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Backend - Mono-repo Entry Point
=======================================

Ponto de entrada principal da API FastAPI para o sistema de precificação.
Mantém compatibilidade com paths anteriores.

Usage:
    uvicorn backend.main:app --reload
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

# Ajustar PYTHONPATH para encontrar o módulo app
import sys, os
backend_root = os.path.dirname(os.path.abspath(__file__))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

# Import da aplicação principal
from app.api.main import app

# Re-export para compatibilidade
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=["backend/app"]
    )