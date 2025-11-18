# 🧭 Fluxograma Completo da Arquitetura

```mermaid
%% =======================
%% ARQUITETURA - VISÃO GERAL
%% =======================
flowchart LR
  subgraph FE[Frontend (Next.js)]
    U[Usuário (UI Web)]
    UP[Upload Planilha (.xlsx)]
    BS[Busca Inteligente (Texto Livre)]
    RENDER[Renderização de Cards\n• EquipmentCard\n• Preço BRL • Badges\n• Indicadores visuais]
    ACOES[Interações do Usuário\n• Add ao carrinho • Seleção múltipla\n• Ver detalhes • Exportar CSV (batch)]
  end

  subgraph BE[Backend (FastAPI)]
    subgraph API[API Routes]
      R1[/api/upload/]
      R2[/api/data/status/]
      R3[/api/smart-search/]
      R4[/api/smart-search/batch/]
    end

    subgraph XLSX[Processamento XLSX]
      P1[Pandas DataFrame]
      P2[Normalização]
      P3[Validação]
    end

    subgraph IDX[Indexação Semântica]
      M1[Sentence Transformer\n(paraphrase-multi)]
      E1[Embeddings (768D)]
      FIDX[FAISS Index\n• IVF (Shards)\n• Flat Index]
    end

    subgraph SB[Sistema de Busca]
      Q1[1) Embedar Query\n(Sentence Transformer)]
      Q2[2) Busca Vetorial (FAISS)\n• Cosine • Top-K (5)]
      Q3[3) Re-ranking (Cross-Encoder)\n• score refinado]
      Q4[4) Enriquecimento de Dados\n• Valor unitário • Vida útil (meses)\n• Manutenção (%) • Marca/Modelo]
      Q5[5) Cache de Resultados\n• Query Cache (JSON)\n• Otimização]
    end
  end

  subgraph RESP[Resposta JSON]
    J1["{ resultados: [ { sugeridos, valor_unitario,\nvida_util_meses, manutencao_percent,\nconfianca, ranking, marca } ] }"]
  end

  U --> UP
  U --> BS
  UP --> R1
  BS --> R3
  BS -.batch.-> R4
  R1 --> XLSX
  XLSX -->|DataFrame limpo| IDX
  IDX --> FIDX
  R3 --> Q1 --> Q2 --> Q3 --> Q4 --> Q5 --> J1
  R4 --> Q1
  J1 --> RENDER --> ACOES
  classDef fe fill:#0ea5e9,stroke:#0369a1,color:#fff,stroke-width:1.5px;
  classDef be fill:#16a34a,stroke:#14532d,color:#fff,stroke-width:1.5px;
  classDef sub fill:#22c55e,stroke:#065f46,color:#fff;
  class FE fe
  class BE be
  class API,XLSX,IDX,SB sub
```

---

## 📤 1. Upload de Planilha

```mermaid
sequenceDiagram
  participant U as Usuário
  participant FE as Frontend (Next.js)
  participant API as FastAPI /api/upload
  participant PX as Processamento XLSX (pandas)
  participant IDX as Indexação (ST + FAISS)

  U->>FE: Seleciona .xlsx e confirma upload
  FE->>API: POST /api/upload (arquivo)
  API->>PX: Validar formato, parse pandas
  PX->>PX: Normalizar colunas & validar linhas
  PX->>IDX: Gerar embeddings (768D)
  IDX->>IDX: Criar/atualizar índice FAISS (IVF/Flat)
  IDX-->>API: OK (ids + metadados)
  API-->>FE: 200 { status: "indexado" }
  FE-->>U: Feedback de sucesso
```

---

## 🔍 2. Busca Individual

```mermaid
sequenceDiagram
  participant U as Usuário
  participant FE as Frontend
  participant API as FastAPI /api/smart-search
  participant ENC as Embeddings (ST)
  participant FAI as FAISS
  participant RR as Re-ranking (Cross-Encoder)
  participant ENR as Enriquecimento
  participant C as Cache (JSON)

  U->>FE: Digita consulta (texto livre)
  FE->>API: GET /api/smart-search?q=...
  API->>C: Verifica cache (HIT?)
  alt Cache HIT
    C-->>API: resultados cacheados
    API-->>FE: JSON ordenado
  else Cache MISS
    API->>ENC: Embedar query (768D)
    ENC-->>API: vetor consulta
    API->>FAI: top_k por similaridade (cosine)
    FAI-->>API: candidatos K
    API->>RR: Re-ranking candidatos
    RR-->>API: scores refinados + ordem
    API->>ENR: Enriquecer campos (valor, vida útil, manutenção, marca)
    ENR-->>API: itens enriquecidos
    API->>C: Salvar no cache
    API-->>FE: JSON ordenado
  end
  FE-->>U: Cards com preço BRL, badges de confiança
```

---

## 📦 3. Busca em Lote

```mermaid
sequenceDiagram
  participant U as Usuário
  participant FE as Frontend
  participant API as FastAPI /api/smart-search/batch
  participant WP as Worker Pool (paralelo)
  participant ENC as Embeddings
  participant FAI as FAISS
  participant RR as Re-ranking
  participant ENR as Enriquecimento
  participant C as Cache

  U->>FE: Cola várias descrições (linhas/vírgulas)
  FE->>API: POST /api/smart-search/batch (texto)
  API->>API: Split por linha/vírgula
  API->>WP: Disparar jobs em paralelo por query
  loop Para cada query
    WP->>C: Verifica cache
    alt HIT
      C-->>WP: resultado instantâneo
    else MISS
      WP->>ENC: Embedar query
      ENC-->>WP: vetor
      WP->>FAI: top_k
      FAI-->>WP: candidatos
      WP->>RR: re-ranking
      RR-->>WP: scores
      WP->>ENR: enriquecer dados
      ENR-->>WP: item final
      WP->>C: salvar em cache
    end
  end
  WP-->>API: agregação por descrição
  API-->>FE: JSON com grupos por descrição
  FE-->>U: Tabela/CSV exportável
```

---

## 💾 4. Sistema de Cache

```mermaid
flowchart TD
  Q[Query recebida] --> CH{Cache HIT?}
  CH -- Sim --> R1[Retorna resultado imediato]
  CH -- Não --> P[Pipeline Busca\nEmbedding → FAISS → Re-ranking → Enriquecimento]
  P --> S[Salvar no Cache (JSON)]
  S --> R2[Retorna resultado]
```

---

## 🗂️ Estrutura de Dados

```mermaid
classDiagram
  class XLSXEntrada {
    string Equipamento_Material_Revisado
    float Valor_Unitario
    int Vida_Util_Meses
    float Manutencao_Percent
    string Marca
  }

  class EmbeddingVector {
    float[768] values
  }

  class FAISSIndex {
    EmbeddingVector[] embeddings
    int id
    string texto_original
    float valor_unitario
    int vida_util_meses
    float manutencao_percent
    string marca
  }

  class ApiResultadoItem {
    int ranking
    string sugeridos
    float valor_unitario
    int vida_util_meses
    float manutencao_percent
    float confianca
    string marca
    string link_detalhes
  }

  class ApiResposta {
    ApiResultadoItem[] resultados
  }

  XLSXEntrada "N" --> "1" FAISSIndex : indexação
  EmbeddingVector "N" --> "1" FAISSIndex : armazena
  ApiResultadoItem "N" --> "1" ApiResposta : compõe
```
