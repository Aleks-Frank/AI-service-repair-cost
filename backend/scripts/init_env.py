import secrets
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if target.exists():
        print(".env already exists; no changes made.")
        return
    template = (root / ".env.example").read_text(encoding="utf-8")
    template = template.replace("API_KEY=\n", f"API_KEY={secrets.token_urlsafe(32)}\n")
    template = template.replace(
        "\nBITRIX_OUTBOUND_TOKEN=\n", f"\nBITRIX_OUTBOUND_TOKEN={secrets.token_urlsafe(32)}\n"
    )
    with target.open("x", encoding="utf-8") as file:
        file.write(template)
    target.chmod(0o600)
    print("Created .env with local test secrets. Bitrix delivery is disabled.")


if __name__ == "__main__":
    main()
