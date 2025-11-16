# Backend Python Application

Aplicação FastAPI para sistema de precificação inteligente de produtos.

## Estrutura

```
backend/
├── main.py              # Entry point da API
├── app/                 # Código da aplicação
│   ├── __init__.py
│   ├── api/             # Endpoints FastAPI
│   │   └── main.py      # Aplicação principal
│   ├── ingestao/        # Sistema de ingestão de dados
│   │   └── excel.py     # Processamento de planilhas
│   ├── processamento/   # Processamento de dados e ML
│   │   ├── abbrev_map.json
│   │   ├── attributes.py
│   │   ├── domain_synonyms.json
│   │   ├── normalize.py
│   │   ├── semantic_index.py
│   │   ├── semantic_reranker.py
│   │   ├── similarity.py
│   │   └── smart_search.py
│   └── utils/           # Utilitários
│       ├── config.py
│       ├── data_cache.py
│       ├── db_pool.py
│       ├── db.py
│       └── preferences.py
└── data/                # Cache e índices
    ├── cache/
    └── semantic_index/
```

## Execução

### Desenvolvimento
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Produção
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

- `POST /buscar-inteligente` - Busca semântica de produtos
- `POST /upload` - Upload de planilhas
- `GET /data/status` - Status dos dados
- `POST /feedback` - Sistema de feedback