from pathlib import Path

project = "DI-engine"
author = "OpenDILab"

extensions = [
    "autoapi.extension",
]

templates_path = []
exclude_patterns = ["_build"]

REPO_ROOT = Path(__file__).resolve().parents[2]

autoapi_type = "python"
autoapi_dirs = [
    str(REPO_ROOT / "ding"),
    str(REPO_ROOT / "dizoo"),
]
autoapi_ignore = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
]
autoapi_keep_files = False
autoapi_add_toctree_entry = True
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
    "special-members",
]

html_theme = "alabaster"
