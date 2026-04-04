import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file as a fallback
load_dotenv()

def load_config() -> dict:
    """Loads configuration from oml.yaml, falling back to empty dict."""
    config_path = Path("oml.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to parse oml.yaml: {e}")
    return {}

_config = load_config()


def _parse_bool(value, default: bool = False) -> bool:
    """Parse common string/bool forms into a bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def get_config_val(section: str, key: str, env_var: str, default: str) -> str:
    """Helper to fetch from yaml -> env -> default"""
    if section in _config and key in _config[section]:
        return str(_config[section][key])
    return os.getenv(env_var, default)


def get_config_bool(section: str, key: str, env_var: str, default: bool) -> bool:
    """Helper to fetch a bool from yaml -> env -> default."""
    if section in _config and key in _config[section]:
        return _parse_bool(_config[section][key], default)
    env_val = os.getenv(env_var)
    return _parse_bool(env_val, default)


def _set_env_bool(env_var: str, value: bool) -> None:
    os.environ[env_var] = "1" if value else "0"


def apply_runtime_environment() -> None:
    """
    Apply runtime environment knobs from config so all entrypoints behave the same.
    """
    hf_offline = get_config_bool("huggingface", "offline", "HF_HUB_OFFLINE", False)
    transformers_offline = get_config_bool(
        "huggingface",
        "transformers_offline",
        "TRANSFORMERS_OFFLINE",
        hf_offline,
    )
    datasets_offline = get_config_bool(
        "huggingface",
        "datasets_offline",
        "HF_DATASETS_OFFLINE",
        hf_offline,
    )
    hf_disable_telemetry = get_config_bool(
        "huggingface",
        "disable_telemetry",
        "HF_HUB_DISABLE_TELEMETRY",
        True,
    )
    hf_disable_progress_bars = get_config_bool(
        "huggingface",
        "disable_progress_bars",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        True,
    )
    transformers_verbosity = get_config_val(
        "huggingface",
        "transformers_verbosity",
        "TRANSFORMERS_VERBOSITY",
        "error",
    ).strip().lower()

    _set_env_bool("HF_HUB_OFFLINE", hf_offline)
    _set_env_bool("TRANSFORMERS_OFFLINE", transformers_offline)
    _set_env_bool("HF_DATASETS_OFFLINE", datasets_offline)
    _set_env_bool("HF_HUB_DISABLE_TELEMETRY", hf_disable_telemetry)
    _set_env_bool("HF_HUB_DISABLE_PROGRESS_BARS", hf_disable_progress_bars)
    os.environ["TRANSFORMERS_VERBOSITY"] = transformers_verbosity


apply_runtime_environment()

# Global defaults
DEFAULT_MODEL = get_config_val("llm", "default_model", "OML_MODEL", "ollama:qwen3.5:cloud")
DEFAULT_STORAGE = get_config_val("ingest", "storage", "OML_STORAGE", "sqlite")
DEFAULT_SUMMARIZER = get_config_val("ingest", "summarizer", "OML_SUMMARIZER", "t5")
DEFAULT_GRAPH_MODEL = get_config_val("ingest", "graph_model", "OML_GRAPH_MODEL", "rebel")
DEFAULT_SQLITE_PATH = get_config_val("storage", "sqlite_path", "OML_SQLITE_PATH", "data/oml.db")
DEFAULT_LANCEDB_PATH = get_config_val("storage", "lancedb_path", "OML_LANCEDB_PATH", "data/lancedb")
DEFAULT_EVENTS_DB_PATH = get_config_val("storage", "events_db_path", "OML_EVENTS_DB", "data/oml_events.db")

# TEEG / MemoryEvolver defaults
# Stage 1 model — defaults to the main LLM if not overridden
TEEG_STAGE1_MODEL = get_config_val("teeg", "stage1_model", "OML_TEEG_STAGE1_MODEL", "")
TEEG_SKEPTICISM = float(get_config_val("teeg", "skepticism", "OML_TEEG_SKEPTICISM", "0.5"))
TEEG_ARCHIVE_THRESHOLD = float(
    get_config_val("teeg", "archive_threshold", "OML_TEEG_ARCHIVE_THRESHOLD", "0.15")
)
TEEG_WARM_STORE_DAYS = int(
    get_config_val("teeg", "warm_store_days", "OML_TEEG_WARM_STORE_DAYS", "30")
)
TEEG_PROPAGATION_FACTOR = float(
    get_config_val("teeg", "propagation_factor", "OML_TEEG_PROPAGATION_FACTOR", "0.30")
)
TEEG_INGEST_CANDIDATES = int(
    get_config_val("teeg", "ingest_candidates", "OML_TEEG_INGEST_CANDIDATES", "10")
)
