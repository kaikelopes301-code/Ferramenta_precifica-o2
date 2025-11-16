# 🧪 Suite de Testes - Sistema de Precificação

Este diretório contém a suite completa de testes automatizados para validação do sistema de precificação de equipamentos.

## 📋 Estrutura dos Testes

```
tests/
├── __init__.py                    # Inicialização do pacote
├── conftest.py                    # Fixtures compartilhadas (pytest)
├── test_api_security.py           # ⚠️  Testes de segurança (CRÍTICO)
├── test_api_endpoints.py          # ✅ Testes funcionais de endpoints
├── test_semantic_search.py        # 🧠 Testes de busca semântica
├── test_integration.py            # 🔄 Testes de integração E2E
├── test_performance.py            # ⚡ Testes de performance e carga
└── README.md                      # Este arquivo
```

## 🚀 Como Executar os Testes

### Instalação de Dependências

```bash
# Instalar dependências de teste
pip install -r requirements-test.txt

# Ou instalar todas as dependências
pip install -r requirements.txt
pip install pytest pytest-cov pytest-xdist pytest-timeout psutil
```

### Execução Básica

```bash
# Executar todos os testes
pytest

# Executar testes específicos
pytest tests/test_api_security.py
pytest tests/test_api_endpoints.py

# Executar com cobertura
pytest --cov=backend --cov-report=html

# Executar em paralelo (mais rápido)
pytest -n auto
```

### Execução por Categoria

```bash
# Testes de segurança (PRIORIDADE ALTA)
pytest -m security

# Testes de API
pytest -m api

# Testes de busca semântica
pytest -m semantic

# Testes de integração
pytest -m integration

# Testes de performance
pytest -m performance

# Testes de carga (demorados)
pytest -m load

# Testes de stress
pytest -m stress
```

### Execução Seletiva

```bash
# Pular testes lentos
pytest -m "not slow"

# Apenas testes rápidos
pytest -m "not slow and not load and not stress"

# Testes de segurança + API
pytest -m "security or api"

# Testes críticos para produção
pytest -m "security or integration"
```

### Execução com Verbosidade

```bash
# Modo verboso
pytest -v

# Modo extra verboso (mostra prints)
pytest -vv -s

# Mostra duração dos testes
pytest --durations=10

# Mostra cobertura por arquivo
pytest --cov=backend --cov-report=term-missing
```

## 📊 Categorias de Testes

### 1. Testes de Segurança (`test_api_security.py`)

**CRÍTICO - DEVE EXECUTAR ANTES DE PRODUÇÃO**

- ❌ SQL Injection
- ❌ Autenticação e Autorização
- ❌ Rate Limiting
- ❌ Validação de Input
- ❌ CORS
- ❌ Vazamento de Secrets

**Executar:**
```bash
pytest tests/test_api_security.py -v
```

**Critério de Aprovação:** Todos os testes devem passar antes de deploy.

### 2. Testes de Endpoints (`test_api_endpoints.py`)

Testes funcionais de todos os endpoints REST:

- ✅ Health Check
- ✅ Upload de Dados
- ✅ Busca Tradicional (TF-IDF)
- ✅ Busca Inteligente (Semântica)
- ✅ Busca em Lote
- ✅ Favoritos
- ✅ Kit e Orçamento
- ✅ Histórico
- ✅ Feedback
- ✅ Detalhes de Equipamento
- ✅ Cache

**Executar:**
```bash
pytest tests/test_api_endpoints.py -v
```

**Cobertura Esperada:** 90%+ dos endpoints

### 3. Testes de Busca Semântica (`test_semantic_search.py`)

Testes dos componentes de IA e busca:

- 🧠 Normalização de Texto
- 🧠 Índice FAISS
- 🧠 Cross-Encoder Reranking
- 🧠 Extração de Atributos
- 🧠 TF-IDF Híbrido
- 🧠 Cache de Índices

**Executar:**
```bash
pytest tests/test_semantic_search.py -v
```

**Nota:** Testes marcados com `@pytest.mark.slow` carregam modelos de IA.

### 4. Testes de Integração (`test_integration.py`)

Testes E2E de fluxos completos:

- 🔄 Upload → Busca → Resultado
- 🔄 Favoritos e Kit
- 🔄 Feedback Loop
- 🔄 Histórico de Buscas
- 🔄 Cache Integration
- 🔄 Cenários Realistas de Usuário

**Executar:**
```bash
pytest tests/test_integration.py -v
```

### 5. Testes de Performance (`test_performance.py`)

Testes de latência, throughput e carga:

- ⚡ Latência de Endpoints
- ⚡ Throughput e Concorrência
- ⚡ Memory Leaks
- ⚡ Cache Performance
- ⚡ Load Testing (30s)
- ⚡ Stress Testing (100 req)

**Executar:**
```bash
# Todos os testes de performance
pytest tests/test_performance.py -v

# Apenas latência e throughput
pytest tests/test_performance.py -k "latency or throughput" -v

# Load e stress (DEMORADOS)
pytest tests/test_performance.py -m "load or stress" -v
```

**Benchmarks Esperados:**
- Health Check: <100ms
- Busca Tradicional: <2s
- Busca Inteligente (primeira): <30s
- Busca Inteligente (cached): <5s
- Throughput: >50 req/s (health), >3 req/s (search)

## 🎯 Checklist Pré-Produção

Antes de fazer deploy em produção, execute:

```bash
# 1. Testes de segurança (OBRIGATÓRIO)
pytest -m security -v
# ✅ TODOS devem passar

# 2. Testes de API (OBRIGATÓRIO)
pytest tests/test_api_endpoints.py -v
# ✅ >95% devem passar

# 3. Testes de integração (OBRIGATÓRIO)
pytest tests/test_integration.py -v
# ✅ >90% devem passar

# 4. Smoke test rápido (OPCIONAL)
pytest -m "not slow and not load and not stress" -v
# ✅ Todos devem passar

# 5. Testes de performance (RECOMENDADO)
pytest tests/test_performance.py -k "latency" -v
# ✅ Latências dentro dos limites

# 6. Cobertura de código (RECOMENDADO)
pytest --cov=backend --cov-report=term-missing --cov-fail-under=70
# ✅ Cobertura >= 70%
```

## 📈 Relatórios

### Relatório de Cobertura HTML

```bash
pytest --cov=backend --cov-report=html
# Abre htmlcov/index.html no navegador
```

### Relatório JUnit (para CI/CD)

```bash
pytest --junitxml=reports/junit.xml
```

### Relatório JSON

```bash
pytest --json-report --json-report-file=reports/report.json
```

## 🔧 Configuração (pytest.ini)

Veja `pytest.ini` na raiz do projeto para configurações:

- Markers personalizados
- Diretórios de teste
- Opções padrão
- Warnings

## 🐛 Debugging de Testes

```bash
# Parar no primeiro erro
pytest -x

# Entrar no debugger ao falhar
pytest --pdb

# Executar teste específico
pytest tests/test_api_security.py::TestSQLInjection::test_sql_injection_favoritos_delete

# Verbose com prints
pytest -vv -s tests/test_api_endpoints.py
```

## 📝 Escrevendo Novos Testes

### Estrutura de um Teste

```python
import pytest

class TestMinhaFuncionalidade:
    """Descrição da categoria de testes."""
    
    def test_caso_especifico(self, client, authenticated_headers):
        """Testa comportamento específico."""
        # Arrange (preparar)
        payload = {"campo": "valor"}
        
        # Act (executar)
        response = client.post("/endpoint", json=payload, headers=authenticated_headers)
        
        # Assert (validar)
        assert response.status_code == 200
        assert "campo_esperado" in response.json()
```

### Fixtures Disponíveis (conftest.py)

- `client`: Cliente de teste do FastAPI
- `authenticated_headers`: Headers com `x-user-id`
- `sample_excel_data`: DataFrame de exemplo
- `sample_excel_file`: Arquivo Excel temporário
- `test_db_path`: Banco de dados isolado
- `sample_queries`: Lista de queries de teste
- `performance_metrics`: Coleta de métricas

### Markers Úteis

```python
@pytest.mark.slow  # Teste demorado
@pytest.mark.security  # Teste de segurança
@pytest.mark.api  # Teste de API
@pytest.mark.integration  # Teste de integração
@pytest.mark.performance  # Teste de performance
@pytest.mark.load  # Load testing
@pytest.mark.stress  # Stress testing
@pytest.mark.skip(reason="Motivo")  # Pular teste
@pytest.mark.xfail  # Esperado falhar
```

## 🔍 Troubleshooting

### Erro: "Planilha não encontrada"

```bash
# Certifique-se de que fixtures estão criando dados de teste
pytest --setup-show tests/test_api_endpoints.py
```

### Erro: "Modelos de IA não carregam"

```bash
# Pule testes lentos para debug rápido
pytest -m "not slow"

# Ou aumente timeout
pytest --timeout=60
```

### Erro: "Import error"

```bash
# Verifique PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Testes Lentos Demais

```bash
# Use paralelização
pytest -n 4  # 4 workers

# Pule testes de carga
pytest -m "not load and not stress"
```

## 📚 Recursos Adicionais

- [Documentação Pytest](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pytest Markers](https://docs.pytest.org/en/stable/example/markers.html)
- [Coverage.py](https://coverage.readthedocs.io/)

## ✅ Critérios de Sucesso

Para o sistema estar pronto para produção:

1. ✅ **100%** dos testes de segurança passando
2. ✅ **>95%** dos testes de API passando
3. ✅ **>90%** dos testes de integração passando
4. ✅ **>70%** de cobertura de código
5. ✅ **0** critical bugs nos testes de performance
6. ✅ Latências dentro dos SLAs definidos
7. ✅ Throughput mínimo atendido

---

**Última Atualização:** 10 de novembro de 2025  
**Versão dos Testes:** 1.0.0  
**Status:** ⚠️ Suite completa - EXECUTAR ANTES DE PRODUÇÃO
