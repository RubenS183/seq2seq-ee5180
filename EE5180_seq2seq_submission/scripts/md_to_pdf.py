#!/usr/bin/env python3
"""Render the report Markdown to a print-ready PDF.

Owner: M4. Neither pandoc nor LaTeX is installed on the target machine, so this
goes Markdown -> styled HTML -> PDF via headless Chrome, which every macOS box
running this project already has. Images are inlined as data URIs so the PDF
does not depend on relative paths at print time.

    python scripts/md_to_pdf.py report/EE5180_midterm_report.md
"""
from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: Palatino, Georgia, "Times New Roman", serif; font-size: 10.5pt;
       line-height: 1.5; color: #1a1a1a; max-width: 100%; margin: 0; }
h1 { font-size: 20pt; color: #18204a; margin: 0 0 .3em; line-height: 1.2; }
h2 { font-size: 14pt; color: #18204a; margin: 1.6em 0 .5em; padding-bottom: .18em;
     border-bottom: 1.5px solid #d6dbe8; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #3b4a7a; margin: 1.2em 0 .4em; page-break-after: avoid; }
p { margin: .55em 0; text-align: justify; }
strong { color: #10173a; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt;
       background: #eef1f8; padding: .1em .3em; border-radius: 3px; }
pre { background: #eef1f8; padding: .7em .9em; border-radius: 5px; overflow-x: auto;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; margin: .9em 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th { background: #18204a; color: #fff; text-align: left; padding: .42em .6em; font-weight: 600; }
td { padding: .38em .6em; border-bottom: 1px solid #dfe4ef; vertical-align: top; }
tr:nth-child(even) td { background: #f7f9fc; }
blockquote { margin: 1em 0; padding: .7em 1em; background: #fdf3f3;
             border-radius: 4px; color: #6b2b2b; page-break-inside: avoid; }
blockquote p { margin: .3em 0; text-align: left; }
blockquote strong { color: #98292f; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #d6dbe8; margin: 1.6em 0; }
ul, ol { margin: .5em 0; padding-left: 1.4em; }
li { margin: .28em 0; }
"""


def inline_images(html: str, base: Path) -> str:
    def repl(m):
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = (base / src).resolve()
        if not p.exists():
            print(f"  [warn] missing image: {src}", file=sys.stderr)
            return m.group(0)
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:{mime};base64,{b64}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.markdown).resolve()
    out = Path(args.out) if args.out else src.with_suffix(".pdf")
    if not shutil.which(CHROME) and not Path(CHROME).exists():
        sys.exit(f"Google Chrome not found at {CHROME}; install it or render the .md another way.")

    import markdown

    text = src.read_text()
    # Strip the $$...$$ math delimiters -- there is no MathJax in this pipeline,
    # and a bare formula reads better than stray dollar signs.
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: f"\n> {m.group(1).strip()}\n", text, flags=re.S)
    text = text.replace("\\mathcal{S}", "S").replace("\\sum", "sum").replace("\\log", "log")
    text = re.sub(r"\$(\\mathcal\{U\}|[^$]{1,60}?)\$", lambda m: m.group(1), text)
    text = text.replace("\\mathcal{U}", "U").replace("\\lVert", "||").replace("\\rVert", "||")
    text = text.replace("\\frac{1}{|S|}", "1/|S|").replace("\\qquad", "   ")
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)

    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "attr_list", "sane_lists"]
    )
    html_body = inline_images(html_body, src.parent)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"

    with tempfile.TemporaryDirectory() as td:
        tmp_html = Path(td) / "report.html"
        tmp_html.write_text(html, encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={out}", tmp_html.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
    print(f"[pdf] -> {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
