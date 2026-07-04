import importlib.util
import sys

REQUIRED_PACKAGES = [
    "torch",
    "sklearn",
    "mlflow",
    "dvc",
    "pandas",
    "numpy",
    "pydantic_settings",
]


def check_packages() -> list[str]:
    """
    Verifica se todos os pacotes necessarios estão instalados
    """
    missing = []
    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing


def check_env_file() -> bool:
    """
    verifica se o arquivo env existe
    """
    from pathlib import Path

    return Path(".env").exists()


def check_settings() -> bool:
    """
    Verifica se as configuração estão funcionando
    """
    try:
        from src.settings import settings

        assert settings.random_seed is not None
        return True
    except Exception as e:
        print(f"Erro nas configurações: {e}")


def main() -> None:
    print("=== Validação de Ambiente ===\n")
    ok = True

    missing = check_packages()
    if missing:
        print(f"❌ Pacotes faltando: {missing}")
        ok = False
    else:
        print("✅ Todos os pacotes instalados")

    if check_env_file():
        print("✅ Arquivo .env encontrado")
    else:
        print("❌ Arquivo .env não encontrado — copie o .env.example")
        ok = False

    if check_settings():
        print("✅ Configurações carregadas com sucesso")
    else:
        print("❌ Erro ao carregar configurações")
        ok = False

    print()
    if ok:
        print("Ambiente pronto para uso!")
    else:
        print("Corrija os erros acima antes de continuar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
