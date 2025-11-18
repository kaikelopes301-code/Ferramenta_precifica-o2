import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

EXCEL_PATH = os.getenv("EXCEL_PATH", "../data/dados_internos.xlsx")
EXCEL_SHEET = os.getenv("EXCEL_SHEET", "dados")
SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///../data/app.db")
HTTP_USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "1.5"))
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "../data/feedback.csv")
