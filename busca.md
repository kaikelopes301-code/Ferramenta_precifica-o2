# 🔍 Como Funciona a Busca Inteligente

## O que é?
Busca com **IA** que entende **significado**, não apenas palavras exatas.

**Exemplo:** Busca "vassoura industrial" → Encontra "rodo profissional", "escovão", "varredeira" ✅

---

# 📝 Como Funciona (6 Passos)

# Passo 1: Upload
```
Usuário envia planilha .xlsx → Sistema valida e carrega dados
```

# Passo 2: Normalização
```
Remove acentos, caracteres especiais, padroniza unidades
```

# Passo 3: IA Transforma em Números
```
"vassoura" → [0.23, -0.45, 0.89, ...] (768 números)
Textos similares = números similares
```

# Passo 4: FAISS Busca Rápido
```
Compara 4000 produtos em 50ms
Retorna os 150 mais parecidos
```

# Passo 5: Re-ranking (IA verifica de novo)
```
Cross-Encoder confirma os melhores
Escolhe os 5 top resultados
```

# Passo 6: Cache
```
Salva resultado por 60 segundos
Próxima busca = instantâneo (4ms)
```

#  Tempo Total: 250ms (primeira vez) → 4ms (cache)

#  Tecnologias

**Modelos IA:**
- Sentence Transformers (gera vetores)
- Cross-Encoder (re-ranking)
- FAISS (busca rápida)


