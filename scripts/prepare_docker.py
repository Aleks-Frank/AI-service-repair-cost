"""Подготовить локальные настройки Docker, не перезаписывая существующие .env."""

import os
import secrets
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if not target.exists():
        template = (root / ".env.example").read_text(encoding="utf-8")
        template = template.replace(
            "POSTGRES_PASSWORD=smeta\n", f"POSTGRES_PASSWORD={secrets.token_urlsafe(32)}\n"
        )
        if hasattr(os, "getuid"):
            template += f"\nUID={os.getuid()}\nGID={os.getgid()}\n"
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(template)
        print("Created root .env with a Linux-compatible model cache path.")
    else:
        print("Root .env exists; check HF_CACHE_DIR and UID/GID manually.")
    subprocess.run([sys.executable, str(root / "backend/scripts/init_env.py")], check=True)
    for folder in ("data/hf-cache", "data/import"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    print("Ready. API_KEY is in backend/.env; enter it in the web interface settings.")


if __name__ == "__main__":
    main()
