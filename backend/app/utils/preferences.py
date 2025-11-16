from __future__ import annotations

import json
from typing import Any, Optional
from sqlalchemy import text

from .db import engine


def get_user_preferences(user_id: str) -> dict[str, Any]:
	"""Obtém preferências do usuário armazenadas como JSON.

	Estrutura sugerida:
	  {
		"context_tags": ["Ambiente externo", "Tráfego alto"],
		"shortcuts": {"add_to_kit": true, "simular_troca": true, "gerar_orcamento": true}
	  }
	"""
	with engine.begin() as conn:
		row = conn.execute(text(
			"SELECT data FROM user_preferences WHERE user_id = :u ORDER BY id DESC LIMIT 1"
		), {"u": user_id}).first()
	if not row:
		return {}
	try:
		return json.loads(row[0] or "{}")
	except Exception:
		return {}


def set_user_preferences(user_id: str, data: dict[str, Any]) -> None:
	payload = json.dumps(data or {})
	with engine.begin() as conn:
		conn.execute(text(
			"INSERT INTO user_preferences (user_id, data) VALUES (:u, :d)"
		), {"u": user_id, "d": payload})


def upsert_context_tags(user_id: str, tags: list[str]) -> list[str]:
	prefs = get_user_preferences(user_id)
	prefs["context_tags"] = list(dict.fromkeys([str(t).strip() for t in tags if str(t).strip()]))
	set_user_preferences(user_id, prefs)
	return prefs["context_tags"]

