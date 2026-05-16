from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter


EPS = 1.0e-10


@dataclass(frozen=True)
class AudioConfig:
    channel_order: str
    target_sr: int | None
    start_sec: float
    duration_sec: float | None
    n_fft: int
    hop_length: int
    win_length: int
    energy_percentile: float
    azimuth_bins: int
    elevation_bins: int
    map_gaussian_sigma_deg: float
    beam_smoothing_sigma: float
    aiv_sign: float


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return cleaned[:220] if len(cleaned) > 220 else cleaned


def parse_channel_order(channel_order: str) -> list[str]:
    order = channel_order.replace(",", "").replace(" ", "").upper()
    if len(order) != 4 or set(order) != {"W", "X", "Y", "Z"}:
        raise ValueError(f"channel_order must contain W, X, Y, Z exactly once, got {channel_order!r}")
    return list(order)


def canonicalize_foa(raw_audio: np.ndarray, channel_order: str) -> tuple[np.ndarray, dict[str, Any]]:
    if raw_audio.ndim != 2 or raw_audio.shape[1] < 4:
        raise ValueError(f"FOA audio must be [samples, >=4 channels], got {raw_audio.shape}")
    order = parse_channel_order(channel_order)
    index_by_name = {name: idx for idx, name in enumerate(order)}
    canonical_indices = [index_by_name[name] for name in ("W", "X", "Y", "Z")]
    canonical = raw_audio[:, canonical_indices].astype(np.float32, copy=False)
    return canonical, {
        "input_channel_order": "".join(order),
        "canonical_channel_order": "WXYZ",
        "canonical_indices_from_input": canonical_indices,
    }


def load_audio(path: Path, config: AudioConfig) -> tuple[np.ndarray, int, dict[str, Any]]:
    raw, sr = sf.read(path, always_2d=True)
    raw = raw.astype(np.float32, copy=False)
    original_shape = tuple(raw.shape)
    original_sr = int(sr)

    if config.target_sr is not None and int(config.target_sr) != int(sr):
        gcd = math.gcd(int(sr), int(config.target_sr))
        raw = signal.resample_poly(
            raw,
            up=int(config.target_sr) // gcd,
            down=int(sr) // gcd,
            axis=0,
        ).astype(np.float32, copy=False)
        sr = int(config.target_sr)

    audio, channel_meta = canonicalize_foa(raw, config.channel_order)
    start = max(int(round(float(config.start_sec) * int(sr))), 0)
    if config.duration_sec is None or config.duration_sec <= 0:
        stop = audio.shape[0]
    else:
        stop = min(audio.shape[0], start + int(round(float(config.duration_sec) * int(sr))))
    audio = audio[start:stop]
    if audio.shape[0] < max(config.win_length, 2):
        raise ValueError(f"Audio clip too short after slicing: {audio.shape[0]} samples")

    meta = {
        **channel_meta,
        "path": str(path),
        "original_shape": list(original_shape),
        "original_sample_rate": original_sr,
        "sample_rate": int(sr),
        "clip_start_sec": float(start / max(int(sr), 1)),
        "clip_duration_sec": float(audio.shape[0] / max(int(sr), 1)),
        "clip_samples": int(audio.shape[0]),
    }
    return audio, int(sr), meta


def build_grid(azimuth_bins: int, elevation_bins: int) -> dict[str, np.ndarray]:
    az_edges = np.linspace(-180.0, 180.0, int(azimuth_bins) + 1, dtype=np.float32)
    el_edges = np.linspace(-90.0, 90.0, int(elevation_bins) + 1, dtype=np.float32)
    az_deg = 0.5 * (az_edges[:-1] + az_edges[1:])
    el_deg = 0.5 * (el_edges[:-1] + el_edges[1:])
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    az_mesh, el_mesh = np.meshgrid(az_rad, el_rad, indexing="xy")
    directions = np.stack(
        [
            np.cos(el_mesh) * np.cos(az_mesh),
            np.cos(el_mesh) * np.sin(az_mesh),
            np.sin(el_mesh),
        ],
        axis=-1,
    ).astype(np.float32)
    return {
        "azimuth_deg": az_deg,
        "elevation_deg": el_deg,
        "azimuth_edges_deg": az_edges,
        "elevation_edges_deg": el_edges,
        "directions": directions,
    }


def compute_stft(audio_wxyz: np.ndarray, sr: int, config: AudioConfig) -> tuple[np.ndarray, dict[str, Any]]:
    stfts = []
    for channel in range(4):
        _, times, zxx = signal.stft(
            audio_wxyz[:, channel],
            fs=sr,
            window="hann",
            nperseg=int(config.win_length),
            noverlap=int(config.win_length) - int(config.hop_length),
            nfft=int(config.n_fft),
            boundary=None,
            padded=False,
        )
        stfts.append(zxx.astype(np.complex64))
    stacked = np.stack(stfts, axis=0)
    return stacked, {
        "shape": list(stacked.shape),
        "num_frames": int(stacked.shape[-1]),
        "num_freq_bins": int(stacked.shape[-2]),
        "time_start_sec": float(times[0]) if len(times) else None,
        "time_end_sec": float(times[-1]) if len(times) else None,
    }


def normalize_map(values: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    finite = finite - min(float(np.min(finite)), 0.0)
    max_value = float(np.max(finite)) if finite.size else 0.0
    if max_value <= EPS:
        return np.zeros_like(finite, dtype=np.float32)
    return np.clip(finite / max_value, 0.0, 1.0).astype(np.float32)


def smooth_map(values: np.ndarray, sigma_el: float, sigma_az: float) -> np.ndarray:
    if sigma_el <= 0.0 and sigma_az <= 0.0:
        return values.astype(np.float32, copy=False)
    return gaussian_filter(values, sigma=(float(sigma_el), float(sigma_az)), mode=("nearest", "wrap")).astype(np.float32)


def compute_active_bins(stft_wxyz: np.ndarray, percentile: float) -> np.ndarray:
    power = np.abs(stft_wxyz[0]) ** 2 + np.mean(np.abs(stft_wxyz[1:4]) ** 2, axis=0)
    if power.size == 0:
        return np.zeros(power.shape, dtype=bool)
    threshold = np.percentile(power.reshape(-1), float(percentile))
    mask = power >= max(float(threshold), EPS)
    if not np.any(mask):
        mask = power >= np.max(power)
    return mask


def angles_from_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=-1)
    valid = norms > EPS
    unit = np.zeros_like(vectors, dtype=np.float32)
    unit[valid] = vectors[valid] / norms[valid, None]
    azimuth = np.rad2deg(np.arctan2(unit[:, 1], unit[:, 0]))
    elevation = np.rad2deg(np.arctan2(unit[:, 2], np.maximum(np.linalg.norm(unit[:, :2], axis=-1), EPS)))
    return azimuth.astype(np.float32), elevation.astype(np.float32)


def compute_intensity_map(
    stft_wxyz: np.ndarray,
    active_mask: np.ndarray,
    grid: dict[str, np.ndarray],
    config: AudioConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    w = stft_wxyz[0][active_mask]
    xyz = np.stack([stft_wxyz[1][active_mask], stft_wxyz[2][active_mask], stft_wxyz[3][active_mask]], axis=-1)
    intensity = float(config.aiv_sign) * np.real(np.conj(w)[..., None] * xyz).astype(np.float32)
    magnitude = np.linalg.norm(intensity, axis=-1)
    valid = np.isfinite(magnitude) & (magnitude > EPS)

    hist = np.zeros((len(grid["elevation_deg"]), len(grid["azimuth_deg"])), dtype=np.float32)
    if np.any(valid):
        az, el = angles_from_vectors(intensity[valid])
        hist_raw, _, _ = np.histogram2d(
            el,
            az,
            bins=[grid["elevation_edges_deg"], grid["azimuth_edges_deg"]],
            weights=magnitude[valid],
        )
        hist = hist_raw.astype(np.float32)

    az_bin_width = 360.0 / max(len(grid["azimuth_deg"]), 1)
    el_bin_width = 180.0 / max(len(grid["elevation_deg"]), 1)
    smoothed = smooth_map(
        hist,
        sigma_el=float(config.map_gaussian_sigma_deg) / max(el_bin_width, EPS),
        sigma_az=float(config.map_gaussian_sigma_deg) / max(az_bin_width, EPS),
    )
    out = normalize_map(smoothed)
    peak = peak_from_map(out, grid)
    peak.update(
        {
            "valid_tf_bins": int(np.sum(valid)),
            "active_tf_bins": int(np.sum(active_mask)),
            "raw_nonzero_bins": int(np.count_nonzero(hist > EPS)),
        }
    )
    return out, peak


def compute_beam_map(
    stft_wxyz: np.ndarray,
    active_mask: np.ndarray,
    grid: dict[str, np.ndarray],
    config: AudioConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    w = stft_wxyz[0][active_mask].reshape(-1)
    xyz = np.stack([stft_wxyz[1][active_mask], stft_wxyz[2][active_mask], stft_wxyz[3][active_mask]], axis=-1)
    if w.size == 0:
        beam = np.zeros((len(grid["elevation_deg"]), len(grid["azimuth_deg"])), dtype=np.float32)
        return beam, peak_from_map(beam, grid)

    eww = float(np.mean(np.abs(w) ** 2))
    c = np.real(np.mean(np.conj(w)[:, None] * xyz, axis=0)).astype(np.float32)
    r = np.real(np.einsum("ni,nj->ij", np.conj(xyz), xyz) / max(xyz.shape[0], 1)).astype(np.float32)

    dirs = grid["directions"].reshape(-1, 3)
    cross = 2.0 * (dirs @ c)
    quadratic = np.einsum("ni,ij,nj->n", dirs, r, dirs)
    power = 0.25 * (eww + cross + quadratic)
    power = np.maximum(power, 0.0).reshape(len(grid["elevation_deg"]), len(grid["azimuth_deg"]))
    smoothed = smooth_map(power.astype(np.float32), sigma_el=float(config.beam_smoothing_sigma), sigma_az=float(config.beam_smoothing_sigma))
    out = normalize_map(smoothed)
    peak = peak_from_map(out, grid)
    peak.update({"active_tf_bins": int(np.sum(active_mask)), "eww": eww})
    return out, peak


def peak_from_map(direction_map: np.ndarray, grid: dict[str, np.ndarray]) -> dict[str, Any]:
    if direction_map.size == 0:
        return {"peak_azimuth_deg": None, "peak_elevation_deg": None, "peak_score": 0.0}
    idx = np.unravel_index(int(np.argmax(direction_map)), direction_map.shape)
    return {
        "peak_el_idx": int(idx[0]),
        "peak_az_idx": int(idx[1]),
        "peak_azimuth_deg": float(grid["azimuth_deg"][idx[1]]),
        "peak_elevation_deg": float(grid["elevation_deg"][idx[0]]),
        "peak_score": float(direction_map[idx]),
    }


def circular_error_deg(pred: float | None, target: float | None) -> float | None:
    if pred is None or target is None:
        return None
    return float((float(pred) - float(target) + 180.0) % 360.0 - 180.0)


def render_direction_map(
    direction_map: np.ndarray,
    grid: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    peak: dict[str, Any],
    target_azimuth: float | None,
    target_elevation: float | None,
    cmap: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    im = ax.imshow(
        direction_map,
        origin="lower",
        extent=[-180.0, 180.0, -90.0, 90.0],
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    if target_azimuth is not None and target_elevation is not None:
        ax.scatter([target_azimuth], [target_elevation], marker="x", c="cyan", s=90, linewidths=2.0, label="label")
    if peak.get("peak_azimuth_deg") is not None:
        ax.scatter(
            [peak["peak_azimuth_deg"]],
            [peak["peak_elevation_deg"]],
            marker="o",
            facecolors="none",
            edgecolors="white",
            s=90,
            linewidths=1.8,
            label="peak",
        )
    ax.set_title(title)
    ax.set_xlabel("Azimuth deg, +left (AmbiX mic-local)")
    ax.set_ylabel("Elevation deg")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="normalized score")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_overview(
    intensity_map: np.ndarray,
    beam_map: np.ndarray,
    grid: dict[str, np.ndarray],
    output_path: Path,
    metadata: dict[str, Any],
    intensity_peak: dict[str, Any],
    beam_peak: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    target_az = metadata.get("azimuth_deg")
    target_el = metadata.get("elevation_deg")
    for ax, data, title, peak, cmap in [
        (axes[0], intensity_map, "Intensity Vector Map", intensity_peak, "magma"),
        (axes[1], beam_map, "Beam Power Map", beam_peak, "viridis"),
    ]:
        im = ax.imshow(data, origin="lower", extent=[-180, 180, -90, 90], aspect="auto", cmap=cmap, vmin=0, vmax=1)
        if target_az is not None and target_el is not None:
            ax.scatter([target_az], [target_el], marker="x", c="cyan", s=80, linewidths=2, label="label")
        ax.scatter([peak["peak_azimuth_deg"]], [peak["peak_elevation_deg"]], marker="o", facecolors="none", edgecolors="white", s=80, linewidths=1.8, label="peak")
        ax.set_title(title)
        ax.set_xlabel("Azimuth deg, +left")
        ax.set_ylabel("Elevation deg")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, shrink=0.86)
    fig.suptitle(
        f"{metadata.get('sample_id', '')} | {metadata.get('geometry_los', '')} | "
        f"label az/el=({target_az}, {target_el})",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def iter_manifest_rows(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_manifest_rows(
    dataset_root: Path,
    manifest_path: Path,
    limit: int,
    selection: str,
    sample_id_contains: str | None,
) -> list[dict[str, Any]]:
    selected = []
    for row in iter_manifest_rows(manifest_path):
        if selection != "all" and row.get("geometry_los") != selection:
            continue
        if sample_id_contains and sample_id_contains not in row.get("sample_id", ""):
            continue
        rel_audio = row.get("foa_audio_path") or row.get("audio_path")
        if not rel_audio:
            continue
        audio_path = Path(rel_audio)
        if not audio_path.is_absolute():
            audio_path = dataset_root / audio_path
        if not audio_path.exists():
            continue
        item = dict(row)
        item["resolved_audio_path"] = str(audio_path)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def process_row(row: dict[str, Any], output_root: Path, config: AudioConfig) -> dict[str, Any]:
    sample_id = row.get("sample_id") or Path(row["resolved_audio_path"]).stem
    output_dir = output_root / safe_name(sample_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sr, audio_meta = load_audio(Path(row["resolved_audio_path"]), config)
    stft, stft_meta = compute_stft(audio, sr, config)
    grid = build_grid(config.azimuth_bins, config.elevation_bins)
    active_mask = compute_active_bins(stft, config.energy_percentile)
    intensity_map, intensity_peak = compute_intensity_map(stft, active_mask, grid, config)
    beam_map, beam_peak = compute_beam_map(stft, active_mask, grid, config)

    target_azimuth = row.get("azimuth_deg", row.get("continuous_azimuth_deg"))
    target_elevation = row.get("elevation_deg", row.get("continuous_elevation_deg"))
    if target_azimuth is not None:
        target_azimuth = float(target_azimuth)
    if target_elevation is not None:
        target_elevation = float(target_elevation)

    render_direction_map(
        intensity_map,
        grid,
        output_dir / "05_intensity_vector_direction_map.png",
        "05. FOA Intensity Vector Direction Map",
        intensity_peak,
        target_azimuth,
        target_elevation,
        "magma",
    )
    render_direction_map(
        beam_map,
        grid,
        output_dir / "06_beam_power_direction_map.png",
        "06. FOA Beam Power Direction Map",
        beam_peak,
        target_azimuth,
        target_elevation,
        "viridis",
    )
    render_overview(
        intensity_map,
        beam_map,
        grid,
        output_dir / "audio_maps_overview.png",
        row,
        intensity_peak,
        beam_peak,
    )
    np.save(output_dir / "intensity_vector_map.npy", intensity_map.astype(np.float32))
    np.save(output_dir / "beam_power_map.npy", beam_map.astype(np.float32))

    summary = {
        "sample_id": sample_id,
        "audio_path": row["resolved_audio_path"],
        "geometry_los": row.get("geometry_los"),
        "label": {
            "azimuth_deg": target_azimuth,
            "elevation_deg": target_elevation,
            "azimuth_reference": row.get("azimuth_reference"),
            "azimuth_convention": row.get("azimuth_convention"),
        },
        "audio": audio_meta,
        "stft": stft_meta,
        "config": config.__dict__,
        "active_tf_bins": int(np.sum(active_mask)),
        "intensity_peak": {
            **intensity_peak,
            "azimuth_error_deg": circular_error_deg(intensity_peak.get("peak_azimuth_deg"), target_azimuth),
        },
        "beam_peak": {
            **beam_peak,
            "azimuth_error_deg": circular_error_deg(beam_peak.get("peak_azimuth_deg"), target_azimuth),
        },
        "outputs": {
            "intensity_map_png": str(output_dir / "05_intensity_vector_direction_map.png"),
            "beam_map_png": str(output_dir / "06_beam_power_direction_map.png"),
            "overview_png": str(output_dir / "audio_maps_overview.png"),
            "intensity_map_npy": str(output_dir / "intensity_vector_map.npy"),
            "beam_map_npy": str(output_dir / "beam_power_map.npy"),
        },
    }
    (output_dir / "sample_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="02_pipeline audio-only FOA intensity/beam map smoke runner")
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/yu/Extreme SSD/hm3d_audio_only_100k_ambix"))
    parser.add_argument("--manifest-jsonl", type=Path, default=None)
    parser.add_argument("--sample-audio", type=Path, default=None, help="Run a single wav without manifest selection.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--selection", choices=["all", "gLOS", "gNLOS"], default="all")
    parser.add_argument("--sample-id-contains", default=None)
    parser.add_argument("--channel-order", default="WYZX")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--energy-percentile", type=float, default=70.0)
    parser.add_argument("--azimuth-bins", type=int, default=181)
    parser.add_argument("--elevation-bins", type=int, default=91)
    parser.add_argument("--map-gaussian-sigma-deg", type=float, default=14.0)
    parser.add_argument("--beam-smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--aiv-sign", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = args.manifest_jsonl
    if manifest_path is None:
        manifest_path = dataset_root / "manifests" / "dataset_manifest.jsonl"
    manifest_path = manifest_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = AudioConfig(
        channel_order=args.channel_order,
        target_sr=args.target_sr,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        energy_percentile=args.energy_percentile,
        azimuth_bins=args.azimuth_bins,
        elevation_bins=args.elevation_bins,
        map_gaussian_sigma_deg=args.map_gaussian_sigma_deg,
        beam_smoothing_sigma=args.beam_smoothing_sigma,
        aiv_sign=args.aiv_sign,
    )

    if args.sample_audio is not None:
        rows = [
            {
                "sample_id": args.sample_audio.stem,
                "resolved_audio_path": str(args.sample_audio.expanduser().resolve()),
                "azimuth_reference": "unknown",
                "azimuth_convention": "unknown",
            }
        ]
    else:
        rows = select_manifest_rows(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            limit=max(int(args.limit), 1),
            selection=args.selection,
            sample_id_contains=args.sample_id_contains,
        )

    if not rows:
        raise RuntimeError(f"No audio samples selected from {manifest_path}")

    summaries = []
    failures = []
    for index, row in enumerate(rows):
        print(f"[{index + 1}/{len(rows)}] {row.get('sample_id')} {row.get('geometry_los', '')}")
        try:
            summaries.append(process_row(row, output_root, config))
        except Exception as exc:
            failure = {"sample_id": row.get("sample_id"), "audio_path": row.get("resolved_audio_path"), "error": repr(exc)}
            failures.append(failure)
            print(f"[ERROR] {failure}")

    batch_summary = {
        "num_requested": int(args.limit),
        "num_selected": len(rows),
        "num_processed": len(summaries),
        "num_failures": len(failures),
        "dataset_root": str(dataset_root),
        "manifest_jsonl": str(manifest_path),
        "output_root": str(output_root),
        "config": config.__dict__,
        "runtime_sec": float(time.time() - start),
        "samples": summaries,
        "failures": failures,
    }
    (output_root / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    with (output_root / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] processed={len(summaries)} failures={len(failures)} output={output_root}")


if __name__ == "__main__":
    main()
