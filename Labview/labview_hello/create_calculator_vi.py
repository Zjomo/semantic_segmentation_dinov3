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
    / "fake tool add.vi"
)
OUTPUT_VI = SCRIPT_DIR / "calculator.vi"
STRINGS_FILE = SCRIPT_DIR / "calculator_strings.txt"
EXPORT_LOG = SCRIPT_DIR / "calculator_export.log"
IMPORT_LOG = SCRIPT_DIR / "calculator_import.log"

VI_TITLE = "Research MCP Calculator"
VI_DESCRIPTION = (
    "Deterministic numeric calculator VI for scientific automation via MCP. "
    "Use it for quick arithmetic checks, calibration offsets, parameter sanity "
    "checks, and reproducible test calculations before larger workflows. "
    "This version performs x plus y and returns the numeric result."
)
X_DEFAULT = 2.0
Y_DEFAULT = 3.0
EXPECTED_RESULT = 5.0
X_DESCRIPTION = "First numeric operand for the MCP calculator."
Y_DESCRIPTION = "Second numeric operand for the MCP calculator."
RESULT_DESCRIPTION = (
    "Calculated result for the current calculator operation x plus y."
)
X_TITLE = "operand x"
Y_TITLE = "operand y"
RESULT_TITLE = "calculated result"


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
    text = _replace_control_field(text, "x", (r"<DESC>", r"</DESC>"), X_DESCRIPTION)
    text = _replace_control_field(text, "y", (r"<DESC>", r"</DESC>"), Y_DESCRIPTION)
    text = _replace_control_field(
        text, "x+y", (r"<DESC>", r"</DESC>"), RESULT_DESCRIPTION
    )
    text = _replace_control_field(
        text,
        "x",
        (r'<PART ID=82 order=0 type="[^"]+"><LABEL><STEXT>', r"</STEXT></LABEL></PART>"),
        X_TITLE,
    )
    text = _replace_control_field(
        text,
        "y",
        (r'<PART ID=82 order=0 type="[^"]+"><LABEL><STEXT>', r"</STEXT></LABEL></PART>"),
        Y_TITLE,
    )
    text = _replace_control_field(
        text,
        "x+y",
        (r'<PART ID=82 order=0 type="[^"]+"><LABEL><STEXT>', r"</STEXT></LABEL></PART>"),
        RESULT_TITLE,
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

    vi.SetControlValue("x", X_DEFAULT)
    vi.SetControlValue("y", Y_DEFAULT)
    vi.EditMode = False
    vi_raw.InvokeTypes(
        1017,
        0,
        1,
        (24, 0),
        ((11, 1),),
        False,
    )

    result = vi.GetControlValue("x+y")
    result = vi.GetControlValue("x+y")
    if float(result) != EXPECTED_RESULT:
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
