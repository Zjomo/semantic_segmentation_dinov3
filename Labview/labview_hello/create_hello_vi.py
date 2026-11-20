from __future__ import annotations

import re
from pathlib import Path

import pywintypes
import win32com.client


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
LABVIEW_TEMPLATE = (
    WORKSPACE_ROOT
    / "LabVIEW-MCP-Server-Toolkit"
    / "tests"
    / "fake_tool_say_hello.vi"
)
OUTPUT_VI = SCRIPT_DIR / "hello.vi"
STRINGS_FILE = SCRIPT_DIR / "hello_strings.txt"
EXPORT_LOG = SCRIPT_DIR / "hello_export.log"
IMPORT_LOG = SCRIPT_DIR / "hello_import.log"

VI_TITLE = "Research MCP Echo"
VI_DESCRIPTION = (
    "Deterministic self-test VI for scientific automation via MCP. "
    "Use it before acquisition, analysis, or instrument-control workflows. "
    "Input a short experiment status string. "
    "The output returns exactly the same text unchanged so the LLM can verify "
    "LabVIEW-MCP connectivity and reproducible string IO without side effects."
)
HELLO_TEXT = "hello world"
FORMAT_STRING = "%s"
INPUT_TITLE = "experiment status text"
OUTPUT_TITLE = "validated status text"
INPUT_DESCRIPTION = (
    "Short experiment status text used to validate MCP to LabVIEW string input."
)
OUTPUT_DESCRIPTION = (
    "Exact echoed result used to confirm deterministic LabVIEW output."
)


def _open_vi(app, vi_path: Path):
    vi_raw = app._oleobj_.InvokeTypes(
        2004,
        0,
        1,
        (9, 0),
        ((8, 1), (8, 1), (11, 1), (3, 1)),
        str(vi_path),
        "",
        False,
        0,
    )
    return win32com.client.dynamic.Dispatch(vi_raw), vi_raw


def _replace_once(text: str, pattern: str, repl: str) -> str:
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Could not apply pattern: {pattern}")
    return updated


def _replace_control_field(
    text: str, control_name: str, field_pattern: tuple[str, str], replacement: str
) -> str:
    pattern = (
        rf'(<CONTROL[^>]*name="{re.escape(control_name)}">.*?'
        rf"{field_pattern[0]})"
        rf".*?"
        rf"({field_pattern[1]})"
    )
    return _replace_once(text, pattern, rf"\1{replacement}\2")


def main() -> None:
    if not LABVIEW_TEMPLATE.exists():
        raise FileNotFoundError(f"LabVIEW template not found: {LABVIEW_TEMPLATE}")

    for path in (STRINGS_FILE, EXPORT_LOG, IMPORT_LOG):
        if path.exists():
            path.unlink()

    app = win32com.client.dynamic.Dispatch("LabVIEW.Application")
    version_year = str(getattr(app, "FullVersionYear", ""))
    if version_year != "2021":
        raise RuntimeError(
            f"Expected LabVIEW 2021 automation, got {version_year or 'unknown'}"
        )

    try:
        vi, vi_raw = _open_vi(app, OUTPUT_VI)
    except pywintypes.com_error:
        _, source_vi_raw = _open_vi(app, LABVIEW_TEMPLATE)
        source_vi_raw.InvokeTypes(
            1002,
            0,
            1,
            (24, 0),
            ((8, 1), (11, 1), (11, 1)),
            str(OUTPUT_VI),
            True,
            False,
        )
        vi, vi_raw = _open_vi(app, OUTPUT_VI)
    vi.EditMode = True
    vi_raw.InvokeTypes(
        1000,
        0,
        1,
        (24, 0),
        ((8, 1), (11, 1), (8, 1), (11, 1), (11, 1)),
        str(STRINGS_FILE),
        False,
        str(EXPORT_LOG),
        True,
        True,
    )

    text = STRINGS_FILE.read_text(encoding="mbcs")
    text = _replace_once(text, r"<TITLE>.*?</TITLE>", f"<TITLE>{VI_TITLE}</TITLE>")
    text = _replace_once(
        text,
        r"(<DESC>).*?(</DESC>)",
        rf"\1{VI_DESCRIPTION}\2",
    )
    text = _replace_once(
        text,
        (
            r'(<CONTROL[^>]*name="format string">.*?'
            r"<DEFAULT><STRING>).*?(</STRING></DEFAULT>)"
        ),
        rf"\1{FORMAT_STRING}\2",
    )
    text = _replace_control_field(
        text,
        "input string",
        (r"<DESC>", r"</DESC>"),
        INPUT_DESCRIPTION,
    )
    text = _replace_control_field(
        text,
        "resulting string",
        (r"<DESC>", r"</DESC>"),
        OUTPUT_DESCRIPTION,
    )
    text = _replace_control_field(
        text,
        "input string",
        (
            r'<PART ID=82 order=0 type="[^"]+"><LABEL><STEXT>',
            r"</STEXT></LABEL></PART>",
        ),
        INPUT_TITLE,
    )
    text = _replace_control_field(
        text,
        "resulting string",
        (
            r'<PART ID=82 order=0 type="[^"]+"><LABEL><STEXT>',
            r"</STEXT></LABEL></PART>",
        ),
        OUTPUT_TITLE,
    )
    STRINGS_FILE.write_text(text, encoding="mbcs")

    vi_raw.InvokeTypes(
        1001,
        0,
        1,
        (24, 0),
        ((8, 1), (11, 1), (8, 1)),
        str(STRINGS_FILE),
        False,
        str(IMPORT_LOG),
    )

    vi.SetControlValue("input string", HELLO_TEXT)
    vi.EditMode = False
    vi_raw.InvokeTypes(
        1017,
        0,
        1,
        (24, 0),
        ((11, 1),),
        False,
    )

    result = vi.GetControlValue("resulting string")
    result = vi.GetControlValue("resulting string")
    if result != HELLO_TEXT:
        raise RuntimeError(f"Unexpected VI output: {result!r}")

    vi_raw.InvokeTypes(1011, 0, 1, (24, 0), ())
    vi_raw.InvokeTypes(
        1002,
        0,
        1,
        (24, 0),
        ((8, 1), (11, 1), (11, 1)),
        str(OUTPUT_VI),
        False,
        False,
    )

    print(f"Created: {OUTPUT_VI}")
    print(f"Verified output: {result}")


if __name__ == "__main__":
    main()
