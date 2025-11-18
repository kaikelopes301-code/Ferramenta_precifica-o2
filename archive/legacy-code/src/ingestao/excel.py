import pandas as pd
from ..utils.config import EXCEL_PATH, EXCEL_SHEET
from ..utils.data_cache import data_cache
import re

COLS_SYNONYMS = {
    'fonte': ['bid', 'fonte', 'origem'],
    'fornecedor': ['fornecedor', 'vendor'],
    'marca': ['marca', 'brand', 'fabricante', 'marca_do_equipamento', 'marca_equipamento'],
    'descricao': ['descricao', 'descrição', 'desc'],
    'descricao_saneada': ['descricao saneada', 'descrição saneada', 'desc_saneada'],
    'descricao_padronizada': ['descricao_padronizada', 'descrição padronizada', 'desc_padronizada'],
    'valor_unitario': ['valor unitário', 'valor_unitario', 'preco', 'preço', 'valor'],
    'vida_util_meses': ['vida útil(meses)', 'vida_util(meses)', 'vida util (meses)', 'vida_util_meses', 'vida_util', 'vida (meses)'],
    'manutencao': ['manutenção', 'manutencao', 'manutencao (%)', 'manutenção (%)', 'perc_manutencao']
}

def _normalize_cols(cols):
    norm = []
    for c in cols:
        c = str(c).strip().lower()
        c = c.replace("ã", "a").replace("â", "a").replace("á", "a").replace("é", "e").replace("ê", "e").replace("í", "i").replace("ó", "o").replace("ô", "o").replace("ú", "u").replace("ç", "c")
        c = c.replace("-", "_").replace(" ", "_")
        norm.append(c)
    return norm


def load_excel(path: str = EXCEL_PATH, sheet: str = EXCEL_SHEET) -> pd.DataFrame:
    """Carrega Excel com cache inteligente Parquet.
    
    Ganho estimado: -80% I/O time para reprocessamento.
    Cache baseado em timestamp + tamanho do arquivo fonte.
    """
    def _load_and_process_excel(file_path: str) -> pd.DataFrame:
        """Função interna que faz o processamento real."""
        df = pd.read_excel(file_path, sheet_name=sheet)
        src_cols = _normalize_cols(df.columns)
        df.columns = src_cols

        # constrói mapa dinâmico
        col_map = {}
        for target, synonyms in COLS_SYNONYMS.items():
            for s in synonyms:
                key = s.strip().lower().replace(" ", "_").replace("-", "_")
                key = key.replace("ã", "a").replace("â", "a").replace("á", "a").replace("é", "e").replace("ê", "e").replace("í", "i").replace("ó", "o").replace("ô", "o").replace("ú", "u").replace("ç", "c")
                if key in df.columns:
                    col_map[key] = target
                    break

        df = df.rename(columns=col_map)

        expected = ['fonte','fornecedor','marca','descricao','descricao_saneada','descricao_padronizada','valor_unitario','vida_util_meses','manutencao']
        for col in expected:
            if col not in df.columns:
                df[col] = None

        def _vectorized_to_float(series: pd.Series) -> pd.Series:
            """Conversão vetorizada de strings brasileiras para float.
            
            Muito mais rápido que apply() - processa toda a série de uma vez.
            Trata números em formato brasileiro (vírgula decimal, ponto milhar).
            """
            # Cópia para não modificar original
            s = series.astype(str).copy()
            
            # Remove símbolos comuns (vetorizado)
            s = s.str.replace(r'[R$%\s]', '', regex=True)
            
            # Casos especiais: já numéricos
            numeric_mask = pd.to_numeric(s, errors='coerce').notna()
            
            # Padrão brasileiro: vírgula + ponto (ex: "1.234,56")
            br_pattern_mask = s.str.contains(r'\d+\.\d+,\d+', na=False)
            s.loc[br_pattern_mask] = s.loc[br_pattern_mask].str.replace('.', '').str.replace(',', '.')
            
            # Só vírgula decimal (ex: "123,45")
            comma_only_mask = s.str.contains(',', na=False) & ~s.str.contains(r'\.', na=False)
            s.loc[comma_only_mask] = s.loc[comma_only_mask].str.replace(',', '.')
            
            # Conversão final vetorizada
            return pd.to_numeric(s, errors='coerce')

        # tipos e limpeza VETORIZADA (3x apply() → 3x operações vetoriais)
        if 'valor_unitario' in df.columns:
            df['valor_unitario'] = _vectorized_to_float(df['valor_unitario'])
        if 'vida_util_meses' in df.columns:
            df['vida_util_meses'] = _vectorized_to_float(df['vida_util_meses'])
        if 'manutencao' in df.columns:
            m = _vectorized_to_float(df['manutencao'])
            # Conversão percentual condicional (uma única operação vetorial)
            mask_percent = (m <= 1.0) & m.notna()
            m.loc[mask_percent] *= 100.0
            df['manutencao'] = m

        return df[expected]
    
    # Cache inteligente: usa timestamp + size do arquivo como chave
    return data_cache.get_or_compute(path, _load_and_process_excel, f"sheet_{sheet}")