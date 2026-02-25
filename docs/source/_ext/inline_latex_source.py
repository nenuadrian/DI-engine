from html import escape
from pathlib import Path
from typing import Optional

from sphinx.application import Sphinx


def _module_source_path(app: Sphinx, pagename: str) -> Optional[Path]:
    if not pagename.startswith("autoapi/") or not pagename.endswith("/index"):
        return None
    rel_module = pagename[len("autoapi/") : -len("/index")].strip("/")
    if not rel_module:
        return None

    repo_root = Path(app.confdir).resolve().parents[1]
    module_path = repo_root / rel_module
    py_file = module_path.with_suffix(".py")
    if py_file.is_file():
        return py_file
    init_file = module_path / "__init__.py"
    if init_file.is_file():
        return init_file
    return None


def _render_line(line: str) -> str:
    marker = "# Latex:"
    marker_pos = line.find(marker)
    if marker_pos < 0:
        return escape(line)

    prefix = escape(line[:marker_pos])
    formula = escape(line[marker_pos + len(marker) :].strip())
    return f'{prefix}<span class="latex-inline">\\({formula}\\)</span>'


def _render_source_section(source_path: Path) -> str:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    rendered_lines = []
    for lineno, line in enumerate(lines, start=1):
        rendered = _render_line(line)
        rendered_lines.append(
            f'<span class="source-line"><span class="lineno">{lineno:4d}</span>{rendered}</span>'
        )

    file_label = escape(str(source_path))
    code_html = "\n".join(rendered_lines)
    return (
        '<section class="source-with-latex">'
        '<h2>Full Source Code</h2>'
        f'<p class="source-path">{file_label}</p>'
        '<div class="source-block"><pre>'
        f"{code_html}"
        "</pre></div>"
        "</section>"
    )


def _inject_source_block(app: Sphinx, pagename: str, templatename: str, context: dict, doctree) -> None:
    source_path = _module_source_path(app, pagename)
    if source_path is None:
        return
    body = context.get("body")
    if not isinstance(body, str):
        return
    context["body"] = body + _render_source_section(source_path)


def setup(app: Sphinx) -> dict:
    app.connect("html-page-context", _inject_source_block)
    app.add_css_file("inline_latex_source.css")
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
