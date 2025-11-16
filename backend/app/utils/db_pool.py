"""
Connection pooling otimizado para SQL Server.

Substitui conexões diretas pyodbc por pool gerenciado.
Ganho estimado: -30% latência DB, +200% throughput concurrent.
"""
import sqlalchemy.pool as pool
from sqlalchemy import create_engine, text
from contextlib import contextmanager
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SQLServerPool:
    """Pool de conexões otimizado para SQL Server com pyodbc."""
    
    def __init__(self, connection_string: str):
        """
        Inicializa pool com configurações otimizadas.
        
        Args:
            connection_string: String de conexão SQL Server
        """
        self.engine = create_engine(
            connection_string,
            poolclass=pool.QueuePool,
            pool_size=5,              # Conexões ativas simultâneas
            max_overflow=10,          # Conexões extra sob demanda  
            pool_pre_ping=True,       # Testa conexão antes de usar
            pool_recycle=3600,        # Recicla conexões a cada 1h
            echo=False,               # Set True para debug SQL
            connect_args={
                "timeout": 30,
                "autocommit": True
            }
        )
        logger.info(f"SQL Server pool initialized: {self.engine.pool.size()} + {self.engine.pool.overflow()} connections")
    
    @contextmanager
    def get_connection(self):
        """Context manager para conexões auto-gerenciadas."""
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> list:
        """Executa query SELECT com pool automático."""
        with self.get_connection() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()
    
    def execute_non_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> int:
        """Executa INSERT/UPDATE/DELETE com pool automático."""
        with self.get_connection() as conn:
            result = conn.execute(text(query), params or {})
            return result.rowcount
    
    def bulk_insert(self, table: str, data: list[dict], batch_size: int = 1000):
        """Insert em lote otimizado para grandes volumes."""
        if not data:
            return 0
        
        total_inserted = 0
        with self.get_connection() as conn:
            # Process in batches to avoid memory issues
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # Build bulk INSERT
                columns = list(batch[0].keys())
                placeholders = ", ".join([f":{col}" for col in columns])
                query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                
                conn.execute(text(query), batch)
                total_inserted += len(batch)
        
        logger.info(f"Bulk inserted {total_inserted} rows into {table}")
        return total_inserted
    
    def health_check(self) -> bool:
        """Verifica saúde do pool de conexões."""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_pool_status(self) -> Dict[str, int]:
        """Retorna estatísticas do pool."""
        pool_obj = self.engine.pool
        return {
            "size": pool_obj.size(),
            "checked_in": pool_obj.checkedin(),
            "checked_out": pool_obj.checkedout(),
            "overflow": pool_obj.overflow(),
            "invalid": pool_obj.invalid()
        }


# Singleton global pool (inicializar no startup da app)
_global_pool: Optional[SQLServerPool] = None


def init_db_pool(connection_string: str) -> SQLServerPool:
    """Inicializa pool global de conexões."""
    global _global_pool
    _global_pool = SQLServerPool(connection_string)
    return _global_pool


def get_db_pool() -> SQLServerPool:
    """Retorna pool global inicializado."""
    if _global_pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool() first.")
    return _global_pool


# Exemplo de uso na FastAPI app:
"""
from src.utils.db_pool import init_db_pool, get_db_pool

@app.on_event("startup")
async def startup():
    # Inicializar pool no startup
    init_db_pool("mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+17+for+SQL+Server")

@app.get("/data")
async def get_data():
    pool = get_db_pool() 
    results = pool.execute_query("SELECT * FROM equipments WHERE active = :active", {"active": 1})
    return [dict(row) for row in results]
"""