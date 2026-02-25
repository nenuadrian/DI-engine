import sys
from pathlib import Path

project = "DI-engine"
author = "OpenDILab"

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[1]
sys.path.append(str(SOURCE_DIR / "_ext"))

extensions = [
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "inline_latex_source",
]

templates_path = []
exclude_patterns = ["_build"]

autoapi_type = "python"
autoapi_dirs = [
    str(REPO_ROOT / "ding"),
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
html_static_path = ["_static"]

# Allow MathJax processing inside <pre>/<code> blocks so inline formulas from
# "# Latex:" comments can be rendered in the generated source blocks.
mathjax3_config = {
    "options": {
        "skipHtmlTags": ["script", "noscript", "style", "textarea", "annotation", "annotation-xml"],
    }
}
