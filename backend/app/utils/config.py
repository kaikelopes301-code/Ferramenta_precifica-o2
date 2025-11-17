import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


def _normalize_sqlite_uri(uri: str, repo_root: Path) -> str:
    """
    Garante caminho absoluto/gravável para SQLite e cria diretório se faltar.
    Evita erros "unable to open database file" quando o working dir muda.
    """
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri

    path = Path(uri[len(prefix):])
    if not path.is_absolute():
        path = (repo_root / path).resolve()

    path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{path.as_posix()}"


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SQLITE_PATH = REPO_ROOT / "data" / "app.db"

EXCEL_PATH = os.getenv("EXCEL_PATH", "../data/dados_internos.xlsx")
EXCEL_SHEET = os.getenv("EXCEL_SHEET", "dados")
SQLALCHEMY_DATABASE_URI = _normalize_sqlite_uri(
    os.getenv("SQLALCHEMY_DATABASE_URI", f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"),
    REPO_ROOT,
)
HTTP_USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "1.5"))
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "../data/feedback.csv")
