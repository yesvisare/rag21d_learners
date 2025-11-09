"""
API Key Authentication Manager
Implements secure key generation, hashing, and verification.
"""
import hashlib
import secrets
import time
from datetime import datetime
from typing import Dict, Optional, List
from fastapi import Header, HTTPException, status
import os


class APIKeyManager:
    """Manages API keys with SHA-256 hashing (never stores plaintext keys)."""

    def __init__(self, admin_secret: str = None):
        self.admin_secret = admin_secret or os.getenv("ADMIN_SECRET", "changeme")
        # In-memory store: {key_hash: {"name": str, "created": timestamp, "last_used": timestamp, "request_count": int}}
        self.keys: Dict[str, Dict] = {}

    def generate_key(self, name: str, admin_secret: str) -> str:
        """Generate a new API key (returns plaintext ONCE - caller must save it)."""
        if admin_secret != self.admin_secret:
            raise ValueError("Invalid admin secret")

        # Generate key with prefix
        raw_key = f"rag_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)

        # Store metadata
        self.keys[key_hash] = {
            "name": name,
            "created": time.time(),
            "last_used": None,
            "request_count": 0
        }

        return raw_key  # Only returned once!

    def verify_key(self, raw_key: str) -> bool:
        """Verify API key and update usage stats."""
        if not raw_key or not raw_key.startswith("rag_"):
            return False

        key_hash = self._hash_key(raw_key)

        if key_hash in self.keys:
            # Update usage stats
            self.keys[key_hash]["last_used"] = time.time()
            self.keys[key_hash]["request_count"] += 1
            return True

        return False

    def revoke_key(self, raw_key: str, admin_secret: str) -> bool:
        """Revoke an API key."""
        if admin_secret != self.admin_secret:
            raise ValueError("Invalid admin secret")

        key_hash = self._hash_key(raw_key)
        if key_hash in self.keys:
            del self.keys[key_hash]
            return True
        return False

    def list_keys(self, admin_secret: str) -> List[Dict]:
        """List all keys (without exposing actual keys)."""
        if admin_secret != self.admin_secret:
            raise ValueError("Invalid admin secret")

        return [
            {
                "name": meta["name"],
                "created": datetime.fromtimestamp(meta["created"]).isoformat(),
                "last_used": datetime.fromtimestamp(meta["last_used"]).isoformat() if meta["last_used"] else None,
                "request_count": meta["request_count"]
            }
            for meta in self.keys.values()
        ]

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """Hash key with SHA-256."""
        return hashlib.sha256(raw_key.encode()).hexdigest()


# Global instance
key_manager = APIKeyManager()


# FastAPI dependency
async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """FastAPI dependency to verify API key in header."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    if not key_manager.verify_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    return x_api_key
