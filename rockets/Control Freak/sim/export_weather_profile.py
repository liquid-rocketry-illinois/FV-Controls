import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import rocketpy


CONTROL_FREAK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CONTROL_FREAK_ROOT.parents[1]
DATA_DIR = CONTROL_FREAK_ROOT / "data"

for path in (CONTROL_FREAK_ROOT, PROJECT_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sim.rocketpy_setup import (  # noqa: E402
    launch_date,
    launch_elevation,
    launch_lat,
    launch_lon,
    launch_timezone,
)


DEFAULT_OUTPUT = DATA_DIR / "weather" / "weather_profile.csv"


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-like launch datetime from the command line."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use a datetime like 2026-04-18T12:00:00"
        ) from exc


def env_value(quantity, altitude_asl_m: float) -> float:
    """Evaluate a RocketPy environment quantity at ASL altitude."""
    if callable(quantity):
        return float(quantity(altitude_asl_m))
    if hasattr(quantity, "get_value_opt"):
        return float(quantity.get_value_opt(altitude_asl_m))
    if hasattr(quantity, "get_value"):
        return float(quantity.get_value(altitude_asl_m))
    raise TypeError(f"Unsupported RocketPy environment quantity: {quantity!r}")


def build_environment(date: datetime, timezone: str, model_file: str):
    """Create a RocketPy forecast environment for the launch site."""
    env = rocketpy.Environment(
        latitude=launch_lat,
        longitude=launch_lon,
        elevation=launch_elevation,
        date=date,
        timezone=timezone,
    )
    env.set_atmospheric_model(type="Windy", file=model_file)
    return env


def export_weather_profile(
    output_path: Path,
    min_altitude_agl_m: float,
    max_altitude_agl_m: float,
    step_m: float,
    date: datetime,
    timezone: str,
    model_file: str,
):
    """Export an altitude-indexed forecast table for HAL onboard lookup."""
    if step_m <= 0:
        raise ValueError("step_m must be positive.")
    if max_altitude_agl_m < min_altitude_agl_m:
        raise ValueError("max altitude must be greater than min altitude.")

    env = build_environment(date=date, timezone=timezone, model_file=model_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "altitude_asl_m",
        "altitude_agl_m",
        "density_kg_m3",
        "pressure_pa",
        "temperature_K",
        "gravity_m_s2",
        "wind_x_m_s",
        "wind_y_m_s",
        "speed_of_sound_m_s",
        "dynamic_viscosity_pa_s",
    ]

    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        altitude_agl = min_altitude_agl_m
        while altitude_agl <= max_altitude_agl_m + 1e-9:
            altitude_asl = float(launch_elevation + altitude_agl)
            writer.writerow(
                {
                    "altitude_asl_m": f"{altitude_asl:.3f}",
                    "altitude_agl_m": f"{altitude_agl:.3f}",
                    "density_kg_m3": f"{env_value(env.density, altitude_asl):.9g}",
                    "pressure_pa": f"{env_value(env.pressure, altitude_asl):.9g}",
                    "temperature_K": f"{env_value(env.temperature, altitude_asl):.9g}",
                    "gravity_m_s2": f"{env_value(env.gravity, altitude_asl):.9g}",
                    "wind_x_m_s": f"{env_value(env.wind_velocity_x, altitude_asl):.9g}",
                    "wind_y_m_s": f"{env_value(env.wind_velocity_y, altitude_asl):.9g}",
                    "speed_of_sound_m_s": f"{env_value(env.speed_of_sound, altitude_asl):.9g}",
                    "dynamic_viscosity_pa_s": f"{env_value(env.dynamic_viscosity, altitude_asl):.9g}",
                }
            )
            row_count += 1
            altitude_agl += step_m

    metadata = {
        "source": "RocketPy Windy forecast",
        "model_file": model_file,
        "launch_lat": launch_lat,
        "launch_lon": launch_lon,
        "launch_elevation_m": launch_elevation,
        "launch_date": date.isoformat(),
        "launch_timezone": timezone,
        "min_altitude_agl_m": min_altitude_agl_m,
        "max_altitude_agl_m": max_altitude_agl_m,
        "step_m": step_m,
        "row_count": row_count,
        "lookup_note": "Use altitude_asl_m as the primary onboard interpolation coordinate.",
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path, metadata_path, row_count


def main():
    parser = argparse.ArgumentParser(
        description="Export a RocketPy/Windy weather profile CSV for the HAL flight computer."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-altitude-agl", type=float, default=0.0)
    parser.add_argument("--max-altitude-agl", type=float, default=6000.0)
    parser.add_argument("--step", type=float, default=50.0)
    parser.add_argument("--date", type=parse_datetime, default=launch_date)
    parser.add_argument("--timezone", default=launch_timezone)
    parser.add_argument("--model-file", default="GFS")
    args = parser.parse_args()

    output_path, metadata_path, row_count = export_weather_profile(
        output_path=args.output,
        min_altitude_agl_m=args.min_altitude_agl,
        max_altitude_agl_m=args.max_altitude_agl,
        step_m=args.step,
        date=args.date,
        timezone=args.timezone,
        model_file=args.model_file,
    )
    print(f"Exported {row_count} weather rows to {output_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
