from __future__ import annotations

import math
import subprocess
from hashlib import blake2b
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly

from .config import AudioConfig
from .schemas import AudioRenderResult, MicPose


def discover_dry_audio_files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def select_dry_audio_file(dry_audio_files: list[Path], selection_key: str) -> Path:
    if not dry_audio_files:
        raise FileNotFoundError("No dry audio files were found for rendering.")
    digest = blake2b(selection_key.encode("utf-8"), digest_size=8).hexdigest()
    index = int(digest, 16) % len(dry_audio_files)
    return dry_audio_files[index]


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    sample_rate, data = wavfile.read(path)
    samples = np.asarray(data)
    if np.issubdtype(samples.dtype, np.integer):
        max_abs = float(np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / max_abs
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    return samples, int(sample_rate)


def _read_audio_with_ffmpeg(path: Path) -> Tuple[np.ndarray, int]:
    """
    Decode audio via ffmpeg when Python audio backends are unavailable.
    Returns mono float32 samples and sample rate.
    """
    sample_rate = 48000
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        probe = subprocess.run(
            probe_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        probed = probe.stdout.strip()
        if probed:
            sample_rate = int(float(probed))
    except Exception:
        # Keep default sample rate fallback.
        pass

    decode_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "pipe:1",
    ]
    decode = subprocess.run(
        decode_cmd,
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(decode.stdout, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples from {path}")
    return samples, int(sample_rate)


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    suffix = str(path.suffix).lower()
    if suffix == ".wav":
        try:
            return _read_wav(path)
        except Exception:
            # Fallback to soundfile for uncommon WAV encodings.
            pass

    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        samples = np.asarray(data, dtype=np.float32)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        return samples, int(sample_rate)
    except Exception:
        pass

    try:
        return _read_audio_with_ffmpeg(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read audio file {path}. "
            "Install 'soundfile' or ensure ffmpeg/ffprobe is available for FLAC decoding."
        ) from exc


def _resample_if_needed(samples: np.ndarray, input_sr: int, output_sr: int) -> np.ndarray:
    if int(input_sr) == int(output_sr):
        return samples.astype(np.float32)
    gcd = math.gcd(int(input_sr), int(output_sr))
    up = int(output_sr // gcd)
    down = int(input_sr // gcd)
    return resample_poly(samples, up, down).astype(np.float32)


def _canonicalize_rir(rir: np.ndarray, expected_channels: int) -> np.ndarray:
    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim == 1:
        rir = rir[None, :]
    if rir.shape[0] == expected_channels:
        return rir
    if rir.shape[-1] == expected_channels:
        return rir.T
    if rir.shape[0] < rir.shape[1]:
        return rir
    return rir.T


def remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx(
    rir: np.ndarray,
    yaw_rad: float,
) -> np.ndarray:
    """
    Convert Habitat/RLR 4-channel Ambisonics to mic-local AmbiX WYZX.

    The local build was sanity-rendered as raw RLR N3D channels:
    [W, world +Y, world +Z, world +X]. Habitat camera/mic yaw 0 faces
    world -Z, with +X right and +Y up. This remap rotates the world-axis
    first-order channels into the microphone frame and stores AmbiX/ACN/SN3D
    order [W, Y_left, Z_up, X_front], so audio front matches vision front.
    """
    raw = _canonicalize_rir(rir, expected_channels=4)
    if raw.shape[0] != 4:
        raise ValueError(f"Expected 4-channel RIR for FOA remap, got shape {raw.shape}")

    w_channel = raw[0]
    world_y_up = raw[1]
    world_z = raw[2]
    world_x = raw[3]

    yaw = float(yaw_rad)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    n3d_to_sn3d_order1 = 1.0 / math.sqrt(3.0)

    y_left = (-cos_yaw * world_x + sin_yaw * world_z) * n3d_to_sn3d_order1
    z_up = world_y_up * n3d_to_sn3d_order1
    x_front = (-sin_yaw * world_x - cos_yaw * world_z) * n3d_to_sn3d_order1

    return np.stack([w_channel, y_left, z_up, x_front], axis=0).astype(np.float32)


def _dbfs_to_linear(dbfs: float) -> float:
    return 10.0 ** (float(dbfs) / 20.0)


def _convolve_and_normalize(
    dry_audio: np.ndarray,
    rir: np.ndarray,
    audio_config: AudioConfig,
) -> Tuple[np.ndarray, float, float]:
    channels = []
    for channel_ir in rir:
        rendered = fftconvolve(dry_audio, channel_ir, mode="full").astype(np.float32)
        channels.append(rendered)
    waveform = np.stack(channels, axis=0)

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    target_peak = _dbfs_to_linear(audio_config.normalize_peak_dbfs)
    if peak > 0.0:
        waveform = waveform * (target_peak / peak)
        peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(waveform**2))) if waveform.size else 0.0
    return waveform, peak, rms


def _validate_waveform(
    waveform: np.ndarray,
    peak: float,
    rms: float,
    audio_config: AudioConfig,
) -> str:
    if not np.isfinite(waveform).all():
        return "nan_audio"
    if peak > float(audio_config.clip_peak):
        return "audio_clipped"
    if rms < float(audio_config.silence_rms_threshold):
        return "audio_silent"
    return "success"


def render_spatial_audio(
    session: Any,
    mic_pose: MicPose,
    source_position_world: list[float],
    dry_audio_path: Path,
    layout: dict[str, Path],
    audio_config: AudioConfig,
    *,
    dry_audio_relpath: Optional[str] = None,
    secondary_dry_audio_path: Optional[Path] = None,
    secondary_dry_audio_relpath: Optional[str] = None,
) -> AudioRenderResult:
    dry_audio, dry_sr = _read_audio(dry_audio_path)
    dry_audio = _resample_if_needed(dry_audio, dry_sr, audio_config.sample_rate)
    rir = _canonicalize_rir(
        session.render_rir(source_position_world, mic_pose),
        expected_channels=int(audio_config.channel_count),
    )
    if int(audio_config.channel_count) == 4:
        rir = remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx(rir, mic_pose.yaw_rad)

    rir_npy_path: Optional[Path] = None
    rir_wav_path: Optional[Path] = None

    if not np.isfinite(rir).all():
        return AudioRenderResult(
            rir_generation_status="invalid_rir",
            rendering_status="invalid_rir",
            output_wav_path=None,
            rir_npy_path=None,
            rir_wav_path=None,
            peak_amplitude=None,
            rms=None,
            num_samples=0,
            output_sample_rate=int(audio_config.sample_rate),
            dry_audio_num_samples=int(dry_audio.shape[0]),
            dry_audio_sample_rate=int(audio_config.sample_rate),
            dry_audio_relpath=str(dry_audio_relpath or dry_audio_path),
        )

    if audio_config.write_rir_npy:
        np.save(layout["rir_npy"], rir.astype(np.float32))
        rir_npy_path = layout["rir_npy"]
    if audio_config.write_rir_wav:
        wavfile.write(
            layout["rir_wav"],
            int(audio_config.sample_rate),
            rir.T.astype(np.float32),
        )
        rir_wav_path = layout["rir_wav"]

    waveform, peak, rms = _convolve_and_normalize(dry_audio, rir, audio_config)
    rendering_status = _validate_waveform(waveform, peak, rms, audio_config)
    if rendering_status != "success":
        return AudioRenderResult(
            rir_generation_status="success",
            rendering_status=rendering_status,
            output_wav_path=None,
            rir_npy_path=rir_npy_path,
            rir_wav_path=rir_wav_path,
            peak_amplitude=peak,
            rms=rms,
            num_samples=int(waveform.shape[-1]),
            output_sample_rate=int(audio_config.sample_rate),
            dry_audio_num_samples=int(dry_audio.shape[0]),
            dry_audio_sample_rate=int(audio_config.sample_rate),
            dry_audio_relpath=str(dry_audio_relpath or dry_audio_path),
        )

    wavfile.write(
        layout["audio_wav"],
        int(audio_config.sample_rate),
        waveform.T.astype(np.float32),
    )

    secondary_output_wav_path: Optional[Path] = None
    secondary_rendering_status: Optional[str] = None
    secondary_peak_amplitude: Optional[float] = None
    secondary_rms: Optional[float] = None
    secondary_num_samples = 0
    secondary_dry_audio_num_samples: Optional[int] = None
    secondary_dry_audio_sample_rate: Optional[int] = None

    if secondary_dry_audio_path is not None and audio_config.write_mic_librispeech_wav:
        secondary_dry_audio, secondary_sr = _read_audio(secondary_dry_audio_path)
        secondary_dry_audio = _resample_if_needed(
            secondary_dry_audio,
            secondary_sr,
            audio_config.sample_rate,
        )
        secondary_waveform, secondary_peak, secondary_rms_value = _convolve_and_normalize(
            secondary_dry_audio,
            rir,
            audio_config,
        )
        secondary_status = _validate_waveform(
            secondary_waveform,
            secondary_peak,
            secondary_rms_value,
            audio_config,
        )
        secondary_rendering_status = str(secondary_status)
        secondary_peak_amplitude = float(secondary_peak)
        secondary_rms = float(secondary_rms_value)
        secondary_num_samples = int(secondary_waveform.shape[-1])
        secondary_dry_audio_num_samples = int(secondary_dry_audio.shape[0])
        secondary_dry_audio_sample_rate = int(audio_config.sample_rate)

        if secondary_status != "success":
            return AudioRenderResult(
                rir_generation_status="success",
                rendering_status="success",
                output_wav_path=layout["audio_wav"],
                rir_npy_path=rir_npy_path,
                rir_wav_path=rir_wav_path,
                peak_amplitude=peak,
                rms=rms,
                num_samples=int(waveform.shape[-1]),
                output_sample_rate=int(audio_config.sample_rate),
                dry_audio_num_samples=int(dry_audio.shape[0]),
                dry_audio_sample_rate=int(audio_config.sample_rate),
                dry_audio_relpath=str(dry_audio_relpath or dry_audio_path),
                secondary_output_wav_path=None,
                secondary_rendering_status=secondary_status,
                secondary_peak_amplitude=secondary_peak,
                secondary_rms=secondary_rms_value,
                secondary_num_samples=int(secondary_waveform.shape[-1]),
                secondary_dry_audio_num_samples=secondary_dry_audio_num_samples,
                secondary_dry_audio_sample_rate=int(audio_config.sample_rate),
                secondary_dry_audio_relpath=str(
                    secondary_dry_audio_relpath or secondary_dry_audio_path
                ),
            )

        secondary_output_wav_path = layout["audio_wav_librispeech"]
        wavfile.write(
            secondary_output_wav_path,
            int(audio_config.sample_rate),
            secondary_waveform.T.astype(np.float32),
        )

    return AudioRenderResult(
        rir_generation_status="success",
        rendering_status="success",
        output_wav_path=layout["audio_wav"],
        rir_npy_path=rir_npy_path,
        rir_wav_path=rir_wav_path,
        peak_amplitude=peak,
        rms=rms,
        num_samples=int(waveform.shape[-1]),
        output_sample_rate=int(audio_config.sample_rate),
        dry_audio_num_samples=int(dry_audio.shape[0]),
        dry_audio_sample_rate=int(audio_config.sample_rate),
        dry_audio_relpath=str(dry_audio_relpath or dry_audio_path),
        secondary_output_wav_path=secondary_output_wav_path,
        secondary_rendering_status=secondary_rendering_status,
        secondary_peak_amplitude=secondary_peak_amplitude,
        secondary_rms=secondary_rms,
        secondary_num_samples=secondary_num_samples,
        secondary_dry_audio_num_samples=secondary_dry_audio_num_samples,
        secondary_dry_audio_sample_rate=secondary_dry_audio_sample_rate,
        secondary_dry_audio_relpath=(
            str(secondary_dry_audio_relpath)
            if secondary_dry_audio_relpath is not None
            else None
        ),
    )
