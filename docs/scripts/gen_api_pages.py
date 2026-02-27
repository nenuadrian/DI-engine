from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, List, Tuple

import mkdocs_gen_files

PACKAGE_ROOT = Path("..") / "ding"
IGNORED_PARTS = {"tests", "__pycache__"}
LATEX_MARKER = "# Latex:"


def iter_module_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORED_PARTS for part in rel_parts):
            continue
        if path.name.startswith("test_") or path.name == "__main__.py":
            continue
        yield path


def module_identifier(path: Path, root: Path) -> str:
    module_parts = list(path.relative_to(root).with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    if not module_parts:
        return root.name
    return ".".join([root.name, *module_parts])


def module_doc_path(path: Path, root: Path) -> Path:
    rel_doc_path = path.relative_to(root).with_suffix(".md")
    if path.name == "__init__.py":
        rel_doc_path = rel_doc_path.with_name("index.md")
    return Path("api") / root.name / rel_doc_path


def render_source_line(line: str) -> str:
    marker_pos = line.find(LATEX_MARKER)
    if marker_pos < 0:
        return escape(line)
    prefix = escape(line[:marker_pos])
    formula = escape(line[marker_pos + len(LATEX_MARKER) :].strip())
    return f'{prefix}<span class="latex-inline mathjax-process">\\({formula}\\)</span>'


def render_source_section(source_path: Path) -> str:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    rendered_lines = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        rendered_line = render_source_line(line)
        rendered_lines.append(
            f'<span class="source-line"><span class="lineno">{lineno:4d}</span>{rendered_line}</span>'
        )

    code_html = "".join(rendered_lines)
    source_label = escape(source_path.as_posix())
    return (
        '<section class="source-with-latex">'
        "<h2>Full Source Code</h2>"
        f'<p class="source-path">{source_label}</p>'
        '<div class="source-block"><div class="source-code">'
        f"{code_html}"
        "</div></div>"
        "</section>"
    )


def write_module_page(module_path: Path, root: Path) -> Tuple[str, Path]:
    identifier = module_identifier(module_path, root)
    doc_path = module_doc_path(module_path, root)
    source_section = render_source_section(module_path)

    with mkdocs_gen_files.open(doc_path, "w") as page:
        page.write(f"# `{identifier}`\n\n")
        page.write(f"::: {identifier}\n\n")
        page.write(source_section)
        page.write("\n")

    mkdocs_gen_files.set_edit_path(doc_path, module_path)
    return identifier, doc_path


def write_api_index(module_entries: List[Tuple[str, Path]]) -> None:
    with mkdocs_gen_files.open("api/index.md", "w") as index_file:
        index_file.write("# API Reference\n\n")
        index_file.write("Generated from the `ding` package.\n\n")
        for identifier, doc_path in module_entries:
            link = doc_path.relative_to("api").as_posix()
            index_file.write(f"- [`{identifier}`]({link})\n")


def main() -> None:
    entries: List[Tuple[str, Path]] = []
    for module_path in iter_module_paths(PACKAGE_ROOT):
        entries.append(write_module_page(module_path, PACKAGE_ROOT))
    entries.sort(key=lambda item: item[0])
    write_api_index(entries)


main()
