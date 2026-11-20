from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Calibration:
    reference_shift: float
    slope_a: float
    intercept_b: float
    search_start: float
    search_end: float
    min_snr: float
    axis_file: str = ""
    axis_start: float | None = None
    axis_step: float | None = None
    decimals: int = 4


@dataclass
class PeakResult:
    peak_shift: float
    peak_intensity: float
    baseline: float
    noise: float
    snr: float
    quality: str
    temperature: float


def load_calibration(path: Path) -> Calibration:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return Calibration(
            reference_shift=float(data["reference_shift_cm-1"]),
            slope_a=float(data["temperature_model"]["slope_a"]),
            intercept_b=float(data["temperature_model"]["intercept_b"]),
            search_start=float(data["peak_search"]["start_cm-1"]),
            search_end=float(data["peak_search"]["end_cm-1"]),
            min_snr=float(data["peak_search"]["min_snr"]),
            axis_file=str(data.get("axis", {}).get("axis_file", "")),
            axis_start=data.get("axis", {}).get("axis_start_cm-1"),
            axis_step=data.get("axis", {}).get("axis_step_cm-1"),
            decimals=int(data.get("export", {}).get("decimals", 4)),
        )
    raise ValueError(f"Unsupported calibration format: {path}")


def parse_numeric_series(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p for p in re.split(r"[\s,\t;]+", stripped) if p]
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No numeric data found in {path}")
    return values


def build_axis(
    spectrum_len: int,
    axis_file: Path | None,
    axis_start: float | None,
    axis_step: float | None,
) -> list[float]:
    if axis_file is not None:
        axis = parse_numeric_series(axis_file)
        if len(axis) != spectrum_len:
            raise ValueError(
                f"Axis length {len(axis)} does not match spectrum length {spectrum_len}"
            )
        return axis
    if axis_start is None or axis_step is None:
        raise ValueError(
            "Missing wave number axis. Provide --axis-file or axis_start/axis_step in calibration."
        )
    return [axis_start + axis_step * i for i in range(spectrum_len)]


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def refine_peak_position(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    # Quadratic interpolation around the local maximum.
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0:
        return x1, y1
    offset = 0.5 * (y0 - y2) / denom
    step = x1 - x0
    refined_x = x1 + offset * step
    refined_y = y1 - 0.25 * (y0 - y2) * offset
    return refined_x, refined_y


def classify_quality(snr: float, min_snr: float) -> str:
    if snr >= max(min_snr * 3.0, 10.0):
        return "Good"
    if snr >= min_snr:
        return "Warning"
    return "Bad"


def analyze_spectrum(axis: list[float], spectrum: list[float], calibration: Calibration) -> PeakResult:
    low = min(calibration.search_start, calibration.search_end)
    high = max(calibration.search_start, calibration.search_end)

    in_window = [(x, y, i) for i, (x, y) in enumerate(zip(axis, spectrum)) if low <= x <= high]
    if len(in_window) < 3:
        raise ValueError("Peak search window contains fewer than 3 points.")

    outside = [y for x, y in zip(axis, spectrum) if x < low or x > high]
    if len(outside) < 5:
        edge_count = max(3, min(10, len(in_window) // 4))
        outside = [y for _, y, _ in in_window[:edge_count]] + [y for _, y, _ in in_window[-edge_count:]]

    baseline = median(outside)
    noise = stdev(outside)
    if noise <= 0:
        noise = 1.0

    peak_x, peak_y, peak_i = max(in_window, key=lambda item: item[1])

    if 0 < peak_i < len(spectrum) - 1:
        peak_x, peak_y = refine_peak_position(
            axis[peak_i - 1],
            spectrum[peak_i - 1],
            axis[peak_i],
            spectrum[peak_i],
            axis[peak_i + 1],
            spectrum[peak_i + 1],
        )

    snr = max((peak_y - baseline) / noise, 0.0)
    quality = classify_quality(snr, calibration.min_snr)
    temperature = calibration.slope_a * (peak_x - calibration.reference_shift) + calibration.intercept_b

    return PeakResult(
        peak_shift=peak_x,
        peak_intensity=peak_y,
        baseline=baseline,
        noise=noise,
        snr=snr,
        quality=quality,
        temperature=temperature,
    )


def load_positions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            name = (row.get("filename") or "").strip()
            if name:
                result[name] = row
        return result


def collect_spectra(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.glob(pattern) if p.is_file())


def write_results(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "x_um",
        "y_um",
        "peak_shift_cm-1",
        "peak_intensity",
        "baseline",
        "noise",
        "snr",
        "temperature_c",
        "quality",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Raman spectra into temperature CSV using a calibrated peak-shift model."
    )
    parser.add_argument("--input", required=True, help="Spectrum file or folder.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--calibration", required=True, help="Calibration JSON file.")
    parser.add_argument("--axis-file", default="", help="Optional wave number axis file.")
    parser.add_argument("--positions", default="", help="Optional positions CSV with columns filename,x_um,y_um.")
    parser.add_argument("--pattern", default="*.xls", help="Glob when --input is a folder.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    calibration_path = Path(args.calibration)
    calibration = load_calibration(calibration_path)

    axis_file: Path | None = None
    if args.axis_file:
        axis_file = Path(args.axis_file)
    elif calibration.axis_file:
        axis_file = Path(calibration.axis_file)

    positions = load_positions(Path(args.positions) if args.positions else None)
    spectra = collect_spectra(input_path, args.pattern)
    if not spectra:
        raise FileNotFoundError(f"No spectra found for {input_path} with pattern {args.pattern}")

    rows: list[dict[str, object]] = []
    for spectrum_path in spectra:
        spectrum = parse_numeric_series(spectrum_path)
        axis = build_axis(
            len(spectrum),
            axis_file,
            calibration.axis_start,
            calibration.axis_step,
        )
        result = analyze_spectrum(axis, spectrum, calibration)
        position = positions.get(spectrum_path.name, {})
        decimals = calibration.decimals
        rows.append(
            {
                "filename": spectrum_path.name,
                "x_um": position.get("x_um", ""),
                "y_um": position.get("y_um", ""),
                "peak_shift_cm-1": round(result.peak_shift, decimals),
                "peak_intensity": round(result.peak_intensity, decimals),
                "baseline": round(result.baseline, decimals),
                "noise": round(result.noise, decimals),
                "snr": round(result.snr, decimals),
                "temperature_c": round(result.temperature, decimals),
                "quality": result.quality,
            }
        )

    write_results(rows, output_path)
    print(f"Processed {len(rows)} spectra -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
