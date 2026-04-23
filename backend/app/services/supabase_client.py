"""Non-blocking Supabase client for prediction logging. Skips silently if env vars missing."""

import os
from typing import Optional

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception:
        return None


def log_prediction(payload: dict) -> Optional[dict]:
    client = get_client()
    if client is None:
        return None
    try:
        return client.table("predictions_log").insert(payload).execute().data
    except Exception:
        return None
