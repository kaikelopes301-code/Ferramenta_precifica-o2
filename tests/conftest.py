"""
Fixtures Compartilhadas para Testes
===================================

Configuração centralizada de fixtures do pytest para reutilização.
"""

import pytest
import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
import pandas as pd
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Diretório temporário para dados de teste."""
    temp_dir = tempfile.mkdtemp(prefix="test_precificacao_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_excel_data():
    """DataFrame de exemplo com dados de equipamentos."""
    return pd.DataFrame({
        'fonte': ['bid_001', 'bid_002', 'bid_003', 'bid_004', 'bid_005'],
        'fornecedor': ['Fornecedor A', 'Fornecedor B', 'Fornecedor A', 'Fornecedor C', 'Fornecedor B'],
        'marca': ['Marca X', 'Marca Y', 'Marca X', 'Marca Z', 'Marca Y'],
        'descricao': [
            'MOTOR ELETRICO TRIFASICO 5HP 220V',
            'BOMBA CENTRIFUGA 3HP MONOFASICA',
            'MOTOR ELETRICO TRIFASICO 7.5HP 380V',
            'COMPRESSOR AR 10HP TRIFASICO',
            'BOMBA CENTRIFUGA 5HP TRIFASICA 220V'
        ],
        'descricao_saneada': [
            'motor eletrico trifasico 5hp 220v',
            'bomba centrifuga 3hp monofasica',
            'motor eletrico trifasico 7.5hp 380v',
            'compressor ar 10hp trifasico',
            'bomba centrifuga 5hp trifasica 220v'
        ],
        'descricao_padronizada': [
            'motor eletrico trifasico 5 hp 220 v',
            'bomba centrifuga 3 hp monofasica',
            'motor eletrico trifasico 7.5 hp 380 v',
            'compressor ar 10 hp trifasico',
            'bomba centrifuga 5 hp trifasica 220 v'
        ],
        'valor_unitario': [1500.00, 800.00, 2200.00, 3500.00, 1200.00],
        'vida_util_meses': [60, 48, 60, 72, 48],
        'manutencao': [5.0, 3.5, 5.0, 6.0, 3.5]
    })


@pytest.fixture(scope="session")
def sample_excel_file(test_data_dir, sample_excel_data):
    """Arquivo Excel de exemplo para testes."""
    excel_path = os.path.join(test_data_dir, "dados_internos.xlsx")
    sample_excel_data.to_excel(excel_path, sheet_name="dados", index=False)
    return excel_path


@pytest.fixture(scope="function")
def test_db_path(test_data_dir):
    """Caminho para banco de dados de teste (novo para cada teste)."""
    import time
    db_path = os.path.join(test_data_dir, f"test_{os.getpid()}.db")
    yield f"sqlite:///{db_path}"
    # Cleanup - espera para Windows liberar o arquivo
    time.sleep(0.1)
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except PermissionError:
        pass  # Ignora se ainda estiver em uso


@pytest.fixture(scope="function")
def client(test_db_path, sample_excel_file, monkeypatch):
    """Cliente de teste do FastAPI com configuração isolada."""
    # Configura variáveis de ambiente para teste
    monkeypatch.setenv("EXCEL_PATH", sample_excel_file)
    monkeypatch.setenv("SQLALCHEMY_DATABASE_URI", test_db_path)
    monkeypatch.setenv("FEEDBACK_PATH", os.path.join(os.path.dirname(sample_excel_file), "feedback.csv"))
    
    # Importa app após configurar env vars
    from backend.main import app
    from backend.app.utils.db import init_db
    
    # Inicializa banco de dados de teste
    init_db()
    
    # Cria cliente de teste
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def authenticated_headers():
    """Headers com autenticação de teste."""
    return {
        "x-user-id": "test_user_001",
        "Content-Type": "application/json"
    }


@pytest.fixture(scope="session")
def sample_queries():
    """Queries de teste para busca."""
    return [
        "motor eletrico",
        "bomba",
        "compressor",
        "motor 5hp",
        "bomba centrifuga trifasica",
        "motor trifásico 220v",  # com acento
        "moto eletrico",  # typo intencional
        "vassoura",  # não existe
        "123456",  # número
        ""  # vazio
    ]


@pytest.fixture(scope="function")
def mock_semantic_index(monkeypatch):
    """Mock do índice semântico para testes unitários rápidos."""
    class MockSemanticIndex:
        def search(self, query: str, top_k: int = 10):
            # Retorna índices fixos para teste determinístico
            if "motor" in query.lower():
                return [(0, 0.95), (2, 0.85)]
            elif "bomba" in query.lower():
                return [(1, 0.92), (4, 0.88)]
            elif "compressor" in query.lower():
                return [(3, 0.90)]
            return []
    
    def mock_build_index(*args, **kwargs):
        return MockSemanticIndex()
    
    # Não aplicar monkeypatch se não for teste unitário
    # (permite testes de integração usarem índice real)
    return mock_build_index


@pytest.fixture(scope="function")
def performance_metrics():
    """Dicionário para coletar métricas de performance durante testes."""
    return {
        "latencies": [],
        "memory_usage": [],
        "cache_hits": 0,
        "cache_misses": 0
    }
