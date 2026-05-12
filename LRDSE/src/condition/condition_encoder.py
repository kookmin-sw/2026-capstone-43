import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from src.audio.preprocess import (
    AudioPreprocessConfig,
    crop_or_pad_for_train,
    ensure_mono_2d,
    normalize_noisy,
    stft,
)
from src.condition.preprocess import (
    find_anchor_file,
    find_lowstate_file,
    load_condition_source_cached,
    sample_to_time_from_anchors,
)

try:
    from dataset import resolve_manifest_path
except Exception:  # pragma: no cover - fallback for package-only imports
    def resolve_manifest_path(path_str: str) -> str:
        return path_str


FORCE_FEATURE_NAMES = ("mean", "max", "std", "p95", "dmean", "dmax_abs")


@dataclass
class ConditionFrameFeatureConfig:
    sample_rate: int = 16000
    hop_length: int = 128
    num_frames: int = 256
    raw_force_scale: float = 220.0
    d_force_scale: float = 255.0
    smooth_win: int = 1
    eps: float = 1e-8

    @property
    def feature_dim(self) -> int:
        return 4 * len(FORCE_FEATURE_NAMES)


@dataclass
class NoiseSource:
    audio_path: str
    run_dir: str
    start_sample: int = 0
    end_sample: Optional[int] = None
    channel: str = "0"
    gain: float = 1.0
    condition_sample_offset: int = 0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _resolve_path(value: str) -> str:
    if not value:
        return ""
    return resolve_manifest_path(str(Path(str(value)).expanduser()))


def _path_exists(value: str) -> bool:
    return bool(value) and Path(value).exists()


def _load_json(path: Path) -> Dict:
    import json

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _has_condition_files(path: Path) -> bool:
    try:
        _ = find_lowstate_file(str(path))
        _ = find_anchor_file(str(path))
        return True
    except FileNotFoundError:
        return False


def _candidate_dirs_from_row(row: Dict, meta: Optional[Dict] = None) -> List[Path]:
    meta = meta or {}
    candidates: List[Path] = []

    for key in ("run_dir", "condition_run_dir"):
        value = row.get(key, "") or meta.get(key, "")
        if value:
            candidates.append(Path(_resolve_path(value)))

    for key in ("segment_meta_path",):
        value = row.get(key, "") or meta.get(key, "")
        if value:
            candidates.append(Path(_resolve_path(value)).parent)

    for key in ("noisy_wav", "noisy_audio_path"):
        value = row.get(key, "") or meta.get(key, "")
        if value:
            candidates.append(Path(_resolve_path(value)).parent)

    for key in ("source_dir",):
        value = row.get(key, "") or meta.get(key, "")
        if value:
            candidates.append(Path(_resolve_path(value)))

    for key in ("noise_run_dir",):
        value = row.get(key, "") or meta.get(key, "")
        if value:
            candidates.append(Path(_resolve_path(value)))

    unique = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            unique.append(c)
            seen.add(s)
    return unique


def load_segment_meta_for_row(row: Dict) -> Dict:
    candidates: List[Path] = []

    for key in ("segment_meta_path",):
        value = row.get(key, "")
        if value:
            candidates.append(Path(_resolve_path(value)))

    for key in ("noisy_wav", "noisy_audio_path"):
        value = row.get(key, "")
        if value:
            candidates.append(Path(_resolve_path(value)).parent / "segment_meta.json")

    value = row.get("source_dir", "")
    if value:
        candidates.append(Path(_resolve_path(value)) / "segment_meta.json")

    for path in candidates:
        if path.is_file():
            return _load_json(path)
    return {}


def resolve_condition_run_dir(row: Dict, meta: Optional[Dict] = None) -> str:
    for candidate in _candidate_dirs_from_row(row, meta):
        if candidate.is_dir() and _has_condition_files(candidate):
            return str(candidate)
    raise FileNotFoundError("Could not find a run_dir with lowstate and anchor files.")


def resolve_noise_source(row: Dict, apply_mix_gain: bool = True) -> NoiseSource:
    meta = load_segment_meta_for_row(row)
    merged = {**meta, **row}

    run_dir = resolve_condition_run_dir(row, meta)

    direct_keys = (
        "noise_wav",
        "noise_only_wav",
        "noise_audio_segment_path",
        "noise_only_audio_path",
    )

    audio_path = ""
    start_sample = 0
    end_sample: Optional[int] = None

    for key in direct_keys:
        if merged.get(key, ""):
            audio_path = _resolve_path(merged[key])
            start_sample = 0
            end_sample = None
            break

    if not audio_path and merged.get("noise_audio_path", ""):
        audio_path = _resolve_path(merged["noise_audio_path"])
        start_sample = _safe_int(
            merged.get(
                "noise_start_sample_original",
                merged.get("noise_start_sample_resampled", 0),
            )
        )
        end_value = merged.get(
            "noise_end_sample_original",
            merged.get("noise_end_sample_resampled", ""),
        )
        end_sample = None if end_value == "" else _safe_int(end_value, default=0)

    if not audio_path:
        raise FileNotFoundError(
            "Could not resolve noise-only audio. Provide noise_wav/noise_audio_path, "
            "or use rows whose sample directory contains segment_meta.json."
        )
    if not _path_exists(audio_path):
        raise FileNotFoundError(f"noise audio not found: {audio_path}")

    channel = str(merged.get("noise_channel", "0"))
    gain = 1.0
    if apply_mix_gain:
        gain *= _safe_float(merged.get("noise_scale", 1.0), 1.0)
        gain *= _safe_float(merged.get("peak_gain", 1.0), 1.0)

    run_dir_path = Path(run_dir).resolve()
    condition_sample_offset = 0
    if start_sample > 0 and not (run_dir_path / "segment_meta.json").is_file():
        condition_sample_offset = start_sample

    return NoiseSource(
        audio_path=audio_path,
        run_dir=run_dir,
        start_sample=max(0, int(start_sample)),
        end_sample=end_sample,
        channel=channel,
        gain=float(gain),
        condition_sample_offset=int(condition_sample_offset),
    )


def _to_mono(audio: np.ndarray, channel: str) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)

    if audio.ndim != 2:
        raise ValueError(f"Expected audio [T] or [T,C], got {audio.shape}")

    if str(channel).strip().lower() == "mixdown":
        return audio.mean(axis=1).astype(np.float32)

    ch = _safe_int(channel, default=0)
    if ch < 0 or ch >= audio.shape[1]:
        raise ValueError(f"Invalid channel={ch}, audio has {audio.shape[1]} channels")
    return audio[:, ch].astype(np.float32)


def _resample_if_needed(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32)
    gcd = math.gcd(int(sr), int(target_sr))
    up = int(target_sr) // gcd
    down = int(sr) // gcd
    return resample_poly(wav, up, down).astype(np.float32)


def read_noise_source(
    source: NoiseSource,
    target_sr: int,
) -> Tuple[torch.Tensor, int]:
    frames = -1
    if source.end_sample is not None and source.end_sample > source.start_sample:
        frames = int(source.end_sample) - int(source.start_sample)

    audio, sr = sf.read(
        source.audio_path,
        start=int(source.start_sample),
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    wav = _to_mono(audio, source.channel)
    wav = _resample_if_needed(wav, sr, target_sr)
    wav = wav * float(source.gain)
    return torch.from_numpy(wav).unsqueeze(0).contiguous(), int(target_sr)


def _interp_column(t: np.ndarray, values: np.ndarray, query: float) -> np.ndarray:
    out = np.zeros((values.shape[1],), dtype=np.float64)
    for c in range(values.shape[1]):
        out[c] = np.interp(query, t, values[:, c], left=values[0, c], right=values[-1, c])
    return out


def build_force_frame_features(
    run_dir: str,
    crop_start_sample: int,
    cfg: ConditionFrameFeatureConfig,
) -> torch.Tensor:
    t_low, force_norm, deriv_norm, anchor_sample_idx, anchor_mono_sec = (
        load_condition_source_cached(
            str(Path(run_dir).expanduser().resolve()),
            float(cfg.raw_force_scale),
            float(cfg.d_force_scale),
            int(cfg.smooth_win),
            float(cfg.eps),
        )
    )

    frame_samples = int(crop_start_sample) + (
        np.arange(cfg.num_frames, dtype=np.float64) * float(cfg.hop_length)
    )
    frame_times = np.interp(frame_samples, anchor_sample_idx, anchor_mono_sec)

    if cfg.num_frames == 1:
        half_step = 0.5 * float(cfg.hop_length) / float(cfg.sample_rate)
        edges = np.asarray(
            [frame_times[0] - half_step, frame_times[0] + half_step],
            dtype=np.float64,
        )
    else:
        mids = 0.5 * (frame_times[:-1] + frame_times[1:])
        first_step = max(frame_times[1] - frame_times[0], cfg.eps)
        last_step = max(frame_times[-1] - frame_times[-2], cfg.eps)
        edges = np.empty((cfg.num_frames + 1,), dtype=np.float64)
        edges[1:-1] = mids
        edges[0] = frame_times[0] - 0.5 * first_step
        edges[-1] = frame_times[-1] + 0.5 * last_step

    feats = np.zeros((cfg.feature_dim, cfg.num_frames), dtype=np.float32)

    for ti in range(cfg.num_frames):
        if ti == cfg.num_frames - 1:
            mask = (t_low >= edges[ti]) & (t_low <= edges[ti + 1])
        else:
            mask = (t_low >= edges[ti]) & (t_low < edges[ti + 1])

        if np.any(mask):
            force_win = force_norm[mask]
            deriv_win = deriv_norm[mask]
        else:
            force_win = _interp_column(t_low, force_norm, frame_times[ti])[None, :]
            deriv_win = _interp_column(t_low, deriv_norm, frame_times[ti])[None, :]

        for leg in range(4):
            base = leg * len(FORCE_FEATURE_NAMES)
            f = force_win[:, leg]
            d = deriv_win[:, leg]
            feats[base + 0, ti] = float(np.mean(f))
            feats[base + 1, ti] = float(np.max(f))
            feats[base + 2, ti] = float(np.std(f))
            feats[base + 3, ti] = float(np.quantile(f, 0.95))
            feats[base + 4, ti] = float(np.mean(d))
            feats[base + 5, ti] = float(np.max(np.abs(d)))

    return torch.from_numpy(feats).float()


def build_noise_magnitude_target(
    noise_wave: torch.Tensor,
    cfg: AudioPreprocessConfig,
) -> torch.Tensor:
    spec = stft(ensure_mono_2d(noise_wave), cfg).squeeze(0)
    mag = torch.log1p(torch.abs(spec))

    target_frames = int(cfg.num_frames)
    if mag.size(-1) > target_frames:
        mag = mag[..., :target_frames]
    elif mag.size(-1) < target_frames:
        mag = F.pad(mag, (0, target_frames - mag.size(-1), 0, 0), mode="constant")
    return mag.float()


class ConditionEncoderDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        target_sr: int = 16000,
        target_length: int = 32640,
        n_fft: int = 510,
        hop_length: int = 128,
        num_frames: int = 256,
        center: bool = True,
        normalize_audio: str = "not",
        random_crop: bool = True,
        valid_only: bool = True,
        limit: Optional[int] = None,
        apply_mix_gain: bool = False,
        raw_force_scale: float = 220.0,
        d_force_scale: float = 255.0,
        condition_smooth_win: int = 1,
    ):
        self.manifest_path = Path(manifest_path)
        self.random_crop = bool(random_crop)
        self.normalize_audio = str(normalize_audio).strip().lower()
        self.apply_mix_gain = bool(apply_mix_gain)

        if self.normalize_audio not in {"not", "noise"}:
            raise ValueError("normalize_audio must be one of {'not', 'noise'}")

        self.audio_cfg = AudioPreprocessConfig(
            sample_rate=int(target_sr),
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            num_frames=int(num_frames),
            center=bool(center),
            normalize="not",
        )
        if int(target_length) != self.audio_cfg.train_target_len:
            raise ValueError(
                f"target_length mismatch: got {target_length}, "
                f"expected {self.audio_cfg.train_target_len}"
            )

        self.force_cfg = ConditionFrameFeatureConfig(
            sample_rate=int(target_sr),
            hop_length=int(hop_length),
            num_frames=int(num_frames),
            raw_force_scale=float(raw_force_scale),
            d_force_scale=float(d_force_scale),
            smooth_win=int(condition_smooth_win),
        )

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")

        rows = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if valid_only and "valid" in row and str(row.get("valid", "0")) != "1":
                    continue
                rows.append(dict(row))

        if limit is not None:
            rows = rows[: int(limit)]

        if not rows:
            raise RuntimeError(f"no rows found in manifest: {self.manifest_path}")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        source = resolve_noise_source(row, apply_mix_gain=self.apply_mix_gain)

        noise_wave, _ = read_noise_source(source, target_sr=self.audio_cfg.sample_rate)
        noise_wave, crop_start = crop_or_pad_for_train(
            noise_wave,
            self.audio_cfg,
            start=None,
            random_crop=self.random_crop,
        )

        normfac = torch.tensor(1.0, dtype=torch.float32)
        if self.normalize_audio == "noise":
            noise_wave, normfac = normalize_noisy(noise_wave, self.audio_cfg)

        condition_crop_start = int(source.condition_sample_offset) + int(crop_start)
        force_feat = build_force_frame_features(
            run_dir=source.run_dir,
            crop_start_sample=condition_crop_start,
            cfg=self.force_cfg,
        )
        target_mag = build_noise_magnitude_target(noise_wave, self.audio_cfg)

        return {
            "force_feat": force_feat,
            "target_mag": target_mag,
            "noise_wave": noise_wave.squeeze(0).float(),
            "normfac": normfac.float(),
            "meta": {
                "id": row.get("id", row.get("source_id", str(idx))),
                "run_dir": source.run_dir,
                "noise_audio_path": source.audio_path,
                "crop_start": condition_crop_start,
                "gain": source.gain,
            },
        }


def align_by_delay(
    pred: torch.Tensor,
    target: torch.Tensor,
    delay_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.shape[-1] != target.shape[-1]:
        raise ValueError(f"time length mismatch: {pred.shape} vs {target.shape}")
    t = pred.shape[-1]
    d = int(delay_frames)
    if abs(d) >= t:
        raise ValueError(f"abs(delay_frames) must be < T={t}, got {d}")
    if d > 0:
        return pred[..., : t - d], target[..., d:]
    if d < 0:
        return pred[..., -d:], target[..., : t + d]
    return pred, target


def parse_band_edges_hz(value: str) -> List[float]:
    edges = [float(v.strip()) for v in value.split(",") if v.strip()]
    if len(edges) < 2:
        raise ValueError("band edges must contain at least two values")
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError(f"band edges must be strictly increasing: {edges}")
    return edges


def build_band_slices(
    freq_bins: int,
    sample_rate: int,
    n_fft: int,
    band_edges_hz: Sequence[float],
) -> List[Tuple[int, int]]:
    freqs = np.arange(freq_bins, dtype=np.float64) * float(sample_rate) / float(n_fft)
    slices: List[Tuple[int, int]] = []
    pairs = list(zip(band_edges_hz[:-1], band_edges_hz[1:]))
    for bi, (lo, hi) in enumerate(pairs):
        if bi == len(pairs) - 1:
            idx = np.where((freqs >= float(lo)) & (freqs <= float(hi)))[0]
        else:
            idx = np.where((freqs >= float(lo)) & (freqs < float(hi)))[0]
        if idx.size == 0:
            continue
        start = int(idx[0])
        end = int(idx[-1]) + 1
        if end > start:
            slices.append((start, end))
    if not slices:
        raise ValueError("band edges produced no non-empty frequency bands")
    return slices


def _logmag_to_mag(logmag: torch.Tensor) -> torch.Tensor:
    return torch.expm1(torch.clamp(logmag, min=0.0, max=20.0))


def band_log_energy(
    logmag: torch.Tensor,
    band_slices: Sequence[Tuple[int, int]],
    eps: float = 1e-8,
) -> torch.Tensor:
    mag = _logmag_to_mag(logmag)
    power = mag.pow(2)
    values = []
    for start, end in band_slices:
        values.append(torch.log(power[:, start:end, :].sum(dim=1).clamp_min(eps)))
    return torch.stack(values, dim=1)


def frame_log_energy(logmag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mag = _logmag_to_mag(logmag)
    return torch.log(mag.pow(2).sum(dim=1).clamp_min(eps))


def compute_condition_encoder_losses(
    pred_mag: torch.Tensor,
    target_mag: torch.Tensor,
    delay_frames: int = 0,
    band_slices: Optional[Sequence[Tuple[int, int]]] = None,
    band_weight: float = 0.0,
    event_weight: float = 0.0,
    event_percentile: float = 85.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    pred_aligned, target_aligned = align_by_delay(pred_mag, target_mag, delay_frames)

    l_mag = F.l1_loss(pred_aligned, target_aligned)
    zero = pred_aligned.new_tensor(0.0)
    l_band = zero
    l_event = zero

    if band_weight > 0.0:
        if not band_slices:
            raise ValueError("band_weight > 0 requires band_slices")
        pred_band = band_log_energy(pred_aligned, band_slices, eps=eps)
        target_band = band_log_energy(target_aligned, band_slices, eps=eps)
        l_band = F.l1_loss(pred_band, target_band)

    if event_weight > 0.0:
        target_e = frame_log_energy(target_aligned, eps=eps)
        pred_e = frame_log_energy(pred_aligned, eps=eps)
        q = float(event_percentile) / 100.0
        threshold = torch.quantile(target_e.detach(), q, dim=-1, keepdim=True)
        target_event = (target_e >= threshold).to(dtype=pred_e.dtype)
        pred_logits = pred_e - threshold
        l_event = F.binary_cross_entropy_with_logits(pred_logits, target_event)

    total = l_mag + float(band_weight) * l_band + float(event_weight) * l_event
    return {
        "loss": total,
        "l_mag": l_mag.detach(),
        "l_band": l_band.detach(),
        "l_event": l_event.detach(),
    }


def force_derivative_score(force_feat: torch.Tensor) -> torch.Tensor:
    if force_feat.dim() != 3:
        raise ValueError(f"Expected force_feat [B,24,T], got {tuple(force_feat.shape)}")
    channels = [leg * len(FORCE_FEATURE_NAMES) + 5 for leg in range(4)]
    return force_feat[:, channels, :].abs().sum(dim=1)


def noise_energy_score(target_mag: torch.Tensor) -> torch.Tensor:
    if target_mag.dim() != 3:
        raise ValueError(f"Expected target_mag [B,F,T], got {tuple(target_mag.shape)}")
    return target_mag.sum(dim=1)


def normalized_corr_at_delay(x: torch.Tensor, y: torch.Tensor, delay_frames: int) -> torch.Tensor:
    x_aligned, y_aligned = align_by_delay(x, y, delay_frames)
    x0 = x_aligned - x_aligned.mean(dim=-1, keepdim=True)
    y0 = y_aligned - y_aligned.mean(dim=-1, keepdim=True)
    denom = torch.sqrt((x0.pow(2).sum(dim=-1) * y0.pow(2).sum(dim=-1)).clamp_min(1e-12))
    return (x0 * y0).sum(dim=-1) / denom


@torch.no_grad()
def estimate_delay_from_loader(
    loader,
    min_delay: int = -12,
    max_delay: int = 12,
    num_batches: int = 32,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = torch.device("cpu")

    totals = {d: [] for d in range(int(min_delay), int(max_delay) + 1)}
    count = 0
    for batch in loader:
        force = batch["force_feat"].to(device)
        target = batch["target_mag"].to(device)
        sf = force_derivative_score(force)
        sn = noise_energy_score(target)
        for d in totals:
            totals[d].append(normalized_corr_at_delay(sf, sn, d).detach().cpu())
        count += 1
        if count >= int(num_batches):
            break

    curve = []
    for d, values in totals.items():
        if values:
            corr = torch.cat(values).mean().item()
        else:
            corr = float("nan")
        curve.append({"delay_frames": d, "corr": float(corr)})

    finite = [item for item in curve if math.isfinite(item["corr"])]
    if not finite:
        best_delay = 0
    else:
        best_delay = max(finite, key=lambda item: item["corr"])["delay_frames"]
    return {
        "best_delay_frames": int(best_delay),
        "curve": curve,
        "num_batches": int(count),
    }
