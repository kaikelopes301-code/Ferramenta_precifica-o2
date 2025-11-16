"""
Cache inteligente de DataFrames usando Parquet comprimido.

Substitui operações Excel repetitivas por cache rápido em disco.
Ganho estimado: -80% I/O time para reprocessamento.
"""
import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
import hashlib
import time


class DataCache:
    """Sistema de cache Parquet para acelerar pipelines de dados."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, source_path: str, extra_key: str = "") -> str:
        """Gera chave única baseada no arquivo fonte e timestamp."""
        try:
            mtime = str(os.path.getmtime(source_path))
            size = str(os.path.getsize(source_path))
            base = f"{Path(source_path).stem}_{mtime}_{size}_{extra_key}"
            return hashlib.md5(base.encode()).hexdigest()[:12]
        except (OSError, FileNotFoundError):
            # Fallback para arquivos que não existem fisicamente
            return hashlib.md5(f"{source_path}_{extra_key}_{time.time()}".encode()).hexdigest()[:12]
    
    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Recupera DataFrame do cache se existir."""
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        try:
            if cache_path.exists():
                return pd.read_parquet(cache_path)
        except Exception:
            # Cache corrompido, remove
            cache_path.unlink(missing_ok=True)
        return None
    
    def set(self, cache_key: str, df: pd.DataFrame) -> Path:
        """Salva DataFrame no cache com compressão otimizada."""
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        try:
            df.to_parquet(
                cache_path, 
                compression='snappy',  # Boa compressão + velocidade
                index=False,
                engine='pyarrow'
            )
            return cache_path
        except Exception as e:
            print(f"Warning: Falha ao salvar cache {cache_key}: {e}")
            return cache_path
    
    def get_or_compute(
        self, 
        source_path: str, 
        compute_fn, 
        extra_key: str = "",
        *args, 
        **kwargs
    ) -> pd.DataFrame:
        """Padrão get-or-set: retorna cache ou computa + salva."""
        cache_key = self._get_cache_key(source_path, extra_key)
        
        # Tenta cache primeiro
        cached_df = self.get(cache_key)
        if cached_df is not None:
            return cached_df
            
        # Computa e cacheia
        df = compute_fn(source_path, *args, **kwargs)
        if isinstance(df, pd.DataFrame) and not df.empty:
            self.set(cache_key, df)
        return df
    
    def clear_old_cache(self, max_age_days: int = 7) -> int:
        """Remove arquivos de cache antigos."""
        count = 0
        cutoff = time.time() - (max_age_days * 86400)
        
        for cache_file in self.cache_dir.glob("*.parquet"):
            if cache_file.stat().st_mtime < cutoff:
                cache_file.unlink()
                count += 1
        return count


# Instância global para reutilização
data_cache = DataCache()


def save_intermediate_parquet(df: pd.DataFrame, name: str) -> Path:
    """Helper: salva DataFrame intermediário em Parquet."""
    return data_cache.set(name, df)


def load_intermediate_parquet(name: str) -> Optional[pd.DataFrame]:
    """Helper: carrega DataFrame do cache Parquet."""
    return data_cache.get(name)