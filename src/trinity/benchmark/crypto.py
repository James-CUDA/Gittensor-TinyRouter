"""AES-256-GCM helpers for the hidden TinyRouter benchmark files."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sys
from pathlib import Path

__all__ = ["derive_key", "encrypt_json", "decrypt_json"]


def derive_key(password: str, salt: bytes) -> bytes:
    """PBKDF2-SHA256 key derivation (200k iterations, 32-byte key)."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, dklen=32)


def _require_aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        print("ERROR: cryptography package required. pip install cryptography")
        sys.exit(1)
    return AESGCM


def encrypt_json(data: dict, password: str) -> str:
    """Encrypt a JSON-serializable dict. Returns a base64 string (salt+nonce+ct)."""
    AESGCM = _require_aesgcm()
    plain = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    salt = secrets.token_bytes(16)
    key = derive_key(password, salt)
    nonce = secrets.token_bytes(12)
    ct = AESGCM(key).encrypt(nonce, plain, None)
    return base64.b64encode(salt + nonce + ct).decode("ascii")


def decrypt_json(filepath: Path, password: str) -> dict:
    """Decrypt a benchmark file and return the parsed JSON object."""
    AESGCM = _require_aesgcm()
    combined = base64.b64decode(filepath.read_text().strip())
    salt, nonce, ct = combined[:16], combined[16:28], combined[28:]
    key = derive_key(password, salt)
    plain = AESGCM(key).decrypt(nonce, ct, None)
    return json.loads(plain.decode("utf-8"))
