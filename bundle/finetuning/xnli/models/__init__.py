import importlib

SUPPORTED_SIZES = ("small", "medium", "large_standard", "xl")


def load_model_module(size: str):
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"Unknown model size '{size}'. Supported: {SUPPORTED_SIZES}")
    return importlib.import_module(f"{__name__}.model_{size}")
