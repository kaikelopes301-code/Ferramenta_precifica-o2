from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .config import SQLALCHEMY_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS produtos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fonte TEXT,
                fornecedor TEXT,
                descricao TEXT,
                descricao_saneada TEXT,
                descricao_padronizada TEXT,
                valor_unitario REAL,
                vida_util_meses INTEGER,
                manutencao REAL,
                marca TEXT,
                voltagem TEXT,
                tamanho TEXT,
                atributos JSON
            );
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS precos_externos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                marketplace TEXT,
                produto_ref TEXT,
                titulo TEXT,
                preco REAL,
                url TEXT,
                coletado_em TEXT
            );
            """
        ))

        # Histórico de buscas por usuário
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                context_tags TEXT, -- csv de tags de contexto
                results_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        ))

        # Favoritos dos usuários
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_name TEXT NOT NULL,
                price REAL,
                extra JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        ))

        # Itens do kit por usuário
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS kit_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_name TEXT NOT NULL,
                price REAL,
                qty INTEGER DEFAULT 1,
                extra JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        ))

        # Preferências por usuário (tags de contexto etc.)
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                data JSON,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        ))


    