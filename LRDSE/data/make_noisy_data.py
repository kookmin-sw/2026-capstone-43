#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass
class SourceItem:
    source_dir: Path
    audio_path: Path
    json_path: Optional[Path]
    frames: int
    sr: int


@dataclass
class NoiseRun:
    run_dir: Path
    audio_path: Path
    anchor_path: Path
    lowstate_path: Path
    highstate_path: Path
    frames: int
    sr: int


def parse_source_id(source_id: str):
    parts = source_id.split("-")

    if len(parts) < 3:
        return None, None

    speaker_id = parts[0]
    book_id = parts[1]

    return speaker_id, book_id


def collect_source_items(source_root: Path):
    items = []

    for d in sorted(source_root.iterdir()):
        if not d.is_dir():
            continue

        audio_path = d / "moving_audio.wav"
        json_path = d / "json_data.json"

        if not audio_path.exists():
            continue

        info = sf.info(str(audio_path))

        items.append(
            SourceItem(
                source_dir=d,
                audio_path=audio_path,
                json_path=json_path if json_path.exists() else None,
                frames=info.frames,
                sr=info.samplerate,
            )
        )

    return items


def collect_noise_runs(noise_root: Path):
    runs = []

    for d in sorted(noise_root.iterdir()):
        if not d.is_dir():
            continue

        if d.name == "contaminated":
            continue

        audio_path = d / "audio.wav"
        anchor_path = d / "anchor.json"
        lowstate_path = d / "lowstate.jsonl"
        highstate_path = d / "highstate.jsonl"

        if not audio_path.exists():
            continue
        if not anchor_path.exists():
            continue
        if not lowstate_path.exists():
            continue
        if not highstate_path.exists():
            continue

        info = sf.info(str(audio_path))

        runs.append(
            NoiseRun(
                run_dir=d,
                audio_path=audio_path,
                anchor_path=anchor_path,
                lowstate_path=lowstate_path,
                highstate_path=highstate_path,
                frames=info.frames,
                sr=info.samplerate,
            )
        )

    return runs


def read_audio(path: Path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return wav.astype(np.float32), sr


def to_mono(wav: np.ndarray, channel: Optional[int] = None, mixdown: bool = False):
    if wav.ndim == 1:
        return wav.astype(np.float32)

    if mixdown:
        return wav.mean(axis=1).astype(np.float32)

    if channel is None:
        channel = 0

    if channel < 0 or channel >= wav.shape[1]:
        raise ValueError(f"Invalid channel={channel}, audio has {wav.shape[1]} channels")

    return wav[:, channel].astype(np.float32)


def resample_if_needed(wav: np.ndarray, sr: int, target_sr: int):
    if sr == target_sr:
        return wav.astype(np.float32)

    g = math.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g

    return resample_poly(wav, up, down).astype(np.float32)


def fix_length(wav: np.ndarray, target_len: int):
    if len(wav) == target_len:
        return wav.astype(np.float32)

    if len(wav) > target_len:
        return wav[:target_len].astype(np.float32)

    pad_len = target_len - len(wav)

    if len(wav) == 0:
        return np.zeros(target_len, dtype=np.float32)

    return np.pad(wav, (0, pad_len), mode="edge").astype(np.float32)


def required_noise_len_original(source_len: int, source_sr: int, noise_sr: int):
    return int(math.ceil(source_len * noise_sr / source_sr))


def load_anchor_mapper(anchor_path: Path):
    with anchor_path.open("r", encoding="utf-8") as f:
        anchor = json.load(f)

    total_frames = int(anchor["total_frames"])

    sample_points = []
    time_points = []

    for a in anchor["anchors"]:
        sample = a.get("sample_index_est")

        ns = None
        if "status_htstamp_clock_monotonic_ns" in a:
            ns = a["status_htstamp_clock_monotonic_ns"]
        elif "trigger_htstamp_clock_monotonic_ns" in a:
            ns = a["trigger_htstamp_clock_monotonic_ns"]

        if sample is None or ns is None:
            continue

        sample_points.append(float(sample))
        time_points.append(float(ns))

    if len(sample_points) < 2:
        raise ValueError(f"Not enough valid anchors: {anchor_path}")

    sample_points = np.asarray(sample_points, dtype=np.float64)
    time_points = np.asarray(time_points, dtype=np.float64)

    order = np.argsort(sample_points)
    sample_points = sample_points[order]
    time_points = time_points[order]

    def sample_to_ns(sample_index: int):
        sample_index = max(0, min(int(sample_index), total_frames))
        return int(round(float(np.interp(sample_index, sample_points, time_points))))

    return {
        "total_frames": total_frames,
        "clock_name": anchor.get("clock_name", "UNKNOWN"),
        "run_name": anchor.get("run_name", anchor_path.parent.name),
        "sample_to_ns": sample_to_ns,
    }


def get_anchor_timestamp_ns(anchor_obj: dict):
    if "status_htstamp_clock_monotonic_ns" in anchor_obj:
        return int(anchor_obj["status_htstamp_clock_monotonic_ns"])

    if "trigger_htstamp_clock_monotonic_ns" in anchor_obj:
        return int(anchor_obj["trigger_htstamp_clock_monotonic_ns"])

    return None


def write_anchor_segment(
    src_path: Path,
    dst_path: Path,
    start_ns: int,
    end_ns: int,
    start_sample_original: int,
    end_sample_original: int,
):
    with src_path.open("r", encoding="utf-8") as f:
        anchor = json.load(f)

    segment_anchors = []

    for a in anchor.get("anchors", []):
        t = get_anchor_timestamp_ns(a)

        if t is None:
            continue

        if start_ns <= t < end_ns:
            item = dict(a)
            item["mix_relative_time_sec"] = (t - start_ns) / 1e9

            if "sample_index_est" in item:
                original_sample = int(item["sample_index_est"])
                item["sample_index_est_original"] = original_sample
                item["sample_index_est"] = original_sample - start_sample_original

            segment_anchors.append(item)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    segment = {
        "source_anchor_path": str(src_path),
        "clock_name": anchor.get("clock_name", "UNKNOWN"),
        "run_name": anchor.get("run_name", src_path.parent.name),
        "original_total_frames": anchor.get("total_frames"),
        "segment_total_frames_original": end_sample_original - start_sample_original,
        "segment_start_sample_original": start_sample_original,
        "segment_end_sample_original": end_sample_original,
        "segment_start_clock_monotonic_ns": start_ns,
        "segment_end_clock_monotonic_ns": end_ns,
        "anchors": segment_anchors,
    }

    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(segment, f, indent=2, ensure_ascii=False)

    return len(segment_anchors)


def interval_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int):
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))

    if overlap <= 0:
        return 0.0

    a_len = a_end - a_start
    b_len = b_end - b_start

    if a_len <= 0 or b_len <= 0:
        return 1.0

    return overlap / min(a_len, b_len)


def max_overlap_with_used(start: int, end: int, used_intervals):
    if len(used_intervals) == 0:
        return 0.0

    return max(
        interval_overlap_ratio(start, end, u_start, u_end)
        for u_start, u_end in used_intervals
    )


def is_valid_crop(start: int, end: int, used_intervals, max_overlap_ratio: float):
    return max_overlap_with_used(start, end, used_intervals) <= max_overlap_ratio


def choose_random_noise_crop_with_overlap(
    noise_len: int,
    target_len: int,
    used_intervals,
    rng: random.Random,
    max_overlap_ratio: float = 0.5,
    random_tries: int = 2000,
    grid_step_ratio: float = 0.1,
):
    if noise_len < target_len:
        raise ValueError(
            f"Noise is shorter than source. noise_len={noise_len}, target_len={target_len}"
        )

    max_start = noise_len - target_len

    if max_start <= 0:
        start = 0
        end = target_len

        if is_valid_crop(start, end, used_intervals, max_overlap_ratio):
            return {
                "start_sample": start,
                "end_sample": end,
                "max_overlap_with_previous": max_overlap_with_used(start, end, used_intervals),
            }

        raise ValueError("No valid offset under overlap constraint")

    for _ in range(random_tries):
        start = rng.randint(0, max_start)
        end = start + target_len

        if is_valid_crop(start, end, used_intervals, max_overlap_ratio):
            return {
                "start_sample": start,
                "end_sample": end,
                "max_overlap_with_previous": max_overlap_with_used(start, end, used_intervals),
            }

    step = max(1, int(target_len * grid_step_ratio))
    starts = list(range(0, max_start + 1, step))

    if starts[-1] != max_start:
        starts.append(max_start)

    rng.shuffle(starts)

    for start in starts:
        end = start + target_len

        if is_valid_crop(start, end, used_intervals, max_overlap_ratio):
            return {
                "start_sample": start,
                "end_sample": end,
                "max_overlap_with_previous": max_overlap_with_used(start, end, used_intervals),
            }

    raise ValueError("No valid offset under overlap constraint")


def rms_power(wav: np.ndarray):
    return float(np.mean(wav.astype(np.float64) ** 2))


def scale_noise_to_snr(source: np.ndarray, noise: np.ndarray, snr_db: float):
    source_power = rms_power(source)
    noise_power = rms_power(noise)

    if source_power <= 1e-12:
        raise ValueError("Source audio is almost silent")
    if noise_power <= 1e-12:
        raise ValueError("Noise segment is almost silent")

    snr_linear = 10 ** (snr_db / 10.0)
    scale = math.sqrt(source_power / (noise_power * snr_linear))

    return (noise * scale).astype(np.float32), scale


def peak_normalize_if_needed(wav: np.ndarray, peak_target: float = 0.99):
    peak = float(np.max(np.abs(wav)))

    if peak <= 1e-12:
        return wav.astype(np.float32), 1.0

    if peak <= peak_target:
        return wav.astype(np.float32), 1.0

    gain = peak_target / peak
    return (wav * gain).astype(np.float32), gain


def get_jsonl_timestamp_ns(obj: dict):
    candidates = [
        "clock_monotonic_ns",
        "recv_clock_monotonic_ns",
        "timestamp_ns",
        "time_ns",
    ]

    for key in candidates:
        if key in obj:
            return int(obj[key])

    msg = obj.get("msg")
    if isinstance(msg, dict):
        for key in candidates:
            if key in msg:
                return int(msg[key])

    return None


def write_state_segment(src_path: Path, dst_path: Path, start_ns: int, end_ns: int):
    count = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with src_path.open("r", encoding="utf-8") as fin, dst_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            t = get_jsonl_timestamp_ns(obj)

            if t is None:
                continue

            if start_ns <= t < end_ns:
                obj["mix_relative_time_sec"] = (t - start_ns) / 1e9
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

    return count


def remove_empty_dirs(root: Path):
    removed = 0

    dirs = sorted(
        [p for p in root.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for d in dirs:
        try:
            if not any(d.iterdir()):
                d.rmdir()
                removed += 1
        except OSError:
            continue

    return removed


def make_one_noisy(
    source_item: SourceItem,
    noise_run: NoiseRun,
    snr_db: int,
    rng: random.Random,
    used_intervals_original,
    max_overlap_ratio: float,
    noise_channel: int,
    noise_mixdown: bool,
):
    source_wav, source_sr = read_audio(source_item.audio_path)
    source_wav = to_mono(source_wav, mixdown=True)

    noise_wav, noise_sr = read_audio(noise_run.audio_path)
    noise_wav = to_mono(noise_wav, channel=noise_channel, mixdown=noise_mixdown)

    mapper = load_anchor_mapper(noise_run.anchor_path)

    usable_noise_len = min(len(noise_wav), mapper["total_frames"])

    target_len_original = required_noise_len_original(
        source_len=len(source_wav),
        source_sr=source_sr,
        noise_sr=noise_sr,
    )

    crop_info = choose_random_noise_crop_with_overlap(
        noise_len=usable_noise_len,
        target_len=target_len_original,
        used_intervals=used_intervals_original,
        rng=rng,
        max_overlap_ratio=max_overlap_ratio,
    )

    start_original = crop_info["start_sample"]
    end_original = crop_info["end_sample"]

    noise_crop_original = noise_wav[start_original:end_original].astype(np.float32)

    if noise_sr != source_sr:
        noise_segment = resample_if_needed(noise_crop_original, noise_sr, source_sr)
    else:
        noise_segment = noise_crop_original

    noise_segment = fix_length(noise_segment, len(source_wav))

    scaled_noise, noise_scale = scale_noise_to_snr(source_wav, noise_segment, snr_db)

    noisy = source_wav + scaled_noise
    noisy, peak_gain = peak_normalize_if_needed(noisy, peak_target=0.99)

    start_ns = mapper["sample_to_ns"](start_original)
    end_ns = mapper["sample_to_ns"](end_original)

    start_resampled = int(round(start_original * source_sr / noise_sr))
    end_resampled = start_resampled + len(source_wav)

    meta = {
        "source_dir": str(source_item.source_dir),
        "source_audio_path": str(source_item.audio_path),
        "source_json_path": str(source_item.json_path) if source_item.json_path else "",
        "noise_run_dir": str(noise_run.run_dir),
        "noise_audio_path": str(noise_run.audio_path),
        "noise_anchor_path": str(noise_run.anchor_path),
        "noise_lowstate_path": str(noise_run.lowstate_path),
        "noise_highstate_path": str(noise_run.highstate_path),
        "source_sr": source_sr,
        "noise_sr": noise_sr,
        "duration_sec": len(source_wav) / source_sr,
        "snr_db": snr_db,
        "noise_channel": "mixdown" if noise_mixdown else noise_channel,
        "noise_scale": noise_scale,
        "peak_gain": peak_gain,
        "random_lag_sample_original": start_original,
        "random_lag_sample_resampled": start_resampled,
        "noise_start_sample_original": start_original,
        "noise_end_sample_original": end_original,
        "noise_start_sample_resampled": start_resampled,
        "noise_end_sample_resampled": end_resampled,
        "noise_start_sec_original": start_original / noise_sr,
        "noise_end_sec_original": end_original / noise_sr,
        "noise_start_clock_monotonic_ns": start_ns,
        "noise_end_clock_monotonic_ns": end_ns,
        "noise_overlap_max_ratio": max_overlap_ratio,
        "noise_overlap_with_previous": crop_info["max_overlap_with_previous"],
        "previous_crop_count_same_noise": len(used_intervals_original),
        "anchor_clock_name": mapper["clock_name"],
        "anchor_run_name": mapper["run_name"],
        "anchor_total_frames": mapper["total_frames"],
        "noise_audio_frames": len(noise_wav),
        "usable_noise_frames": usable_noise_len,
    }

    return noisy, meta


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-root",
        type=str,
        default="/home/jaewoo/MAIR/ryu/RIR/sonic_sim/SonicSim/SonicSim-SonicSet/SonicSet/scene_datasets/mp3d_utterance/train_10h",
    )
    parser.add_argument(
        "--noise-root",
        type=str,
        default="/home/jaewoo/Downloads/output/go2_train",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="./noisy",
    )

    parser.add_argument("--snr-min", type=int, default=-5)
    parser.add_argument("--snr-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="몇 개 source마다 진행률을 출력할지 설정",
    )

    parser.add_argument(
        "--max-overlap-ratio",
        type=float,
        default=0.5,
        help="같은 noise run 안에서 crop끼리 허용할 최대 overlap ratio",
    )

    parser.add_argument(
        "--noise-channel",
        type=int,
        default=0,
        help="audio.wav가 multi-channel일 때 사용할 channel index",
    )

    parser.add_argument(
        "--noise-mixdown",
        action="store_true",
        help="audio.wav의 모든 channel을 평균내서 mono로 사용",
    )

    args = parser.parse_args()

    if args.snr_min > args.snr_max:
        raise ValueError("--snr-min must be <= --snr-max")

    progress_every = max(1, args.progress_every)

    source_root = Path(args.source_root).expanduser().resolve()
    noise_root = Path(args.noise_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    rng = random.Random(args.seed)

    source_items = collect_source_items(source_root)
    noise_runs = collect_noise_runs(noise_root)

    if len(source_items) == 0:
        raise RuntimeError(f"No source items found: {source_root}")

    if len(noise_runs) == 0:
        raise RuntimeError(f"No valid noise runs found: {noise_root}")

    rng.shuffle(source_items)

    used_intervals_by_noise = {
        str(noise_run.run_dir): []
        for noise_run in noise_runs
    }

    out_root.mkdir(parents=True, exist_ok=True)
    metadata_path = out_root / "metadata.csv"

    print(f"source items: {len(source_items)}")
    print(f"noise runs: {len(noise_runs)}")
    print(f"SNR integer range: [{args.snr_min}, {args.snr_max}] dB")
    print(f"max overlap ratio: {args.max_overlap_ratio}")
    print(f"progress every: {progress_every}")
    print(f"excluded noise dir: {noise_root / 'contaminated'}")
    print(f"out root: {out_root}")

    fieldnames = [
        "index",
        "speaker_id",
        "book_id",
        "source_id",
        "noise_id",
        "noise_reuse_index",
        "noisy_audio_path",
        "lowstate_segment_path",
        "highstate_segment_path",
        "anchor_segment_path",
        "segment_meta_path",
        "source_dir",
        "source_audio_path",
        "source_json_path",
        "noise_run_dir",
        "noise_audio_path",
        "noise_anchor_path",
        "noise_lowstate_path",
        "noise_highstate_path",
        "source_sr",
        "noise_sr",
        "duration_sec",
        "snr_db",
        "noise_channel",
        "noise_scale",
        "peak_gain",
        "random_lag_sample_original",
        "random_lag_sample_resampled",
        "noise_start_sample_original",
        "noise_end_sample_original",
        "noise_start_sample_resampled",
        "noise_end_sample_resampled",
        "noise_start_sec_original",
        "noise_end_sec_original",
        "noise_start_clock_monotonic_ns",
        "noise_end_clock_monotonic_ns",
        "noise_overlap_max_ratio",
        "noise_overlap_with_previous",
        "previous_crop_count_same_noise",
        "lowstate_segment_count",
        "highstate_segment_count",
        "anchor_segment_count",
        "anchor_clock_name",
        "anchor_run_name",
        "anchor_total_frames",
        "noise_audio_frames",
        "usable_noise_frames",
    ]

    made_count = 0
    skipped_no_valid_noise = 0
    skipped_bad_source_id = 0

    total_sources = len(source_items)
    start_time = time.time()

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for source_idx, source_item in enumerate(source_items, start=1):
            source_id = source_item.source_dir.name
            speaker_id, book_id = parse_source_id(source_id)

            if speaker_id is None or book_id is None:
                skipped_bad_source_id += 1
                if skipped_bad_source_id <= 10:
                    print(f"[SKIP] invalid source_id format: {source_id}")
                continue

            candidate_noise_runs = [
                nr
                for nr in noise_runs
                if nr.frames >= required_noise_len_original(
                    source_len=source_item.frames,
                    source_sr=source_item.sr,
                    noise_sr=nr.sr,
                )
            ]

            rng.shuffle(candidate_noise_runs)

            made_this_source = False
            last_error = None

            for noise_run in candidate_noise_runs:
                noise_id = noise_run.run_dir.name
                noise_key = str(noise_run.run_dir)

                used_intervals_original = used_intervals_by_noise[noise_key]
                noise_reuse_index = len(used_intervals_original) + 1
                snr_db = rng.randint(args.snr_min, args.snr_max)

                sample_out_dir = out_root / speaker_id / book_id / source_id

                noisy_audio_path = sample_out_dir / f"{source_id}.wav"
                lowstate_segment_path = sample_out_dir / "lowstate_segment.jsonl"
                highstate_segment_path = sample_out_dir / "highstate_segment.jsonl"
                anchor_segment_path = sample_out_dir / "anchor_segment.json"
                segment_meta_path = sample_out_dir / "segment_meta.json"

                try:
                    noisy, meta = make_one_noisy(
                        source_item=source_item,
                        noise_run=noise_run,
                        snr_db=snr_db,
                        rng=rng,
                        used_intervals_original=used_intervals_original,
                        max_overlap_ratio=args.max_overlap_ratio,
                        noise_channel=args.noise_channel,
                        noise_mixdown=args.noise_mixdown,
                    )

                    sample_out_dir.mkdir(parents=True, exist_ok=True)

                    sf.write(
                        str(noisy_audio_path),
                        noisy,
                        meta["source_sr"],
                        subtype="PCM_16",
                    )

                    start_ns = meta["noise_start_clock_monotonic_ns"]
                    end_ns = meta["noise_end_clock_monotonic_ns"]

                    low_count = write_state_segment(
                        src_path=noise_run.lowstate_path,
                        dst_path=lowstate_segment_path,
                        start_ns=start_ns,
                        end_ns=end_ns,
                    )

                    high_count = write_state_segment(
                        src_path=noise_run.highstate_path,
                        dst_path=highstate_segment_path,
                        start_ns=start_ns,
                        end_ns=end_ns,
                    )

                    anchor_count = write_anchor_segment(
                        src_path=noise_run.anchor_path,
                        dst_path=anchor_segment_path,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        start_sample_original=meta["noise_start_sample_original"],
                        end_sample_original=meta["noise_end_sample_original"],
                    )

                    used_intervals_original.append(
                        (
                            meta["noise_start_sample_original"],
                            meta["noise_end_sample_original"],
                        )
                    )

                    meta["speaker_id"] = speaker_id
                    meta["book_id"] = book_id
                    meta["source_id"] = source_id
                    meta["noise_id"] = noise_id
                    meta["noise_reuse_index"] = noise_reuse_index
                    meta["lowstate_segment_count"] = low_count
                    meta["highstate_segment_count"] = high_count
                    meta["anchor_segment_count"] = anchor_count
                    meta["noisy_audio_path"] = str(noisy_audio_path)
                    meta["lowstate_segment_path"] = str(lowstate_segment_path)
                    meta["highstate_segment_path"] = str(highstate_segment_path)
                    meta["anchor_segment_path"] = str(anchor_segment_path)
                    meta["segment_meta_path"] = str(segment_meta_path)

                    with segment_meta_path.open("w", encoding="utf-8") as mf:
                        json.dump(meta, mf, indent=2, ensure_ascii=False)

                    writer.writerow(
                        {
                            "index": made_count,
                            **meta,
                        }
                    )

                    made_count += 1
                    made_this_source = True

                    if made_count <= 5 or made_count % progress_every == 0:
                        print(
                            f"[MADE] "
                            f"index={made_count - 1} | "
                            f"source={source_id} | "
                            f"noise={noise_id} | "
                            f"snr={snr_db}dB | "
                            f"low={low_count} | "
                            f"high={high_count} | "
                            f"anchor={anchor_count} | "
                            f"path={noisy_audio_path}"
                        )

                    break

                except Exception as e:
                    last_error = e
                    continue

            if not made_this_source:
                skipped_no_valid_noise += 1
                if skipped_no_valid_noise <= 10:
                    print(f"[SKIP] source={source_item.source_dir} reason={last_error}")

            if (
                source_idx == 1
                or source_idx == total_sources
                or source_idx % progress_every == 0
            ):
                elapsed = time.time() - start_time
                progress = source_idx / total_sources * 100.0
                speed = source_idx / elapsed if elapsed > 0 else 0.0

                remaining = total_sources - source_idx
                eta_sec = remaining / speed if speed > 0 else 0.0

                print(
                    f"[PROGRESS] "
                    f"{source_idx}/{total_sources} "
                    f"({progress:.2f}%) | "
                    f"made={made_count} | "
                    f"skip_invalid_id={skipped_bad_source_id} | "
                    f"skip_no_noise={skipped_no_valid_noise} | "
                    f"elapsed={elapsed/60:.1f}min | "
                    f"eta={eta_sec/60:.1f}min"
                )

    removed_empty_dirs = remove_empty_dirs(out_root)

    print("done")
    print(f"made noisy samples: {made_count}")
    print(f"skipped sources with invalid source_id: {skipped_bad_source_id}")
    print(f"skipped sources with no valid noise offset: {skipped_no_valid_noise}")
    print(f"removed empty dirs: {removed_empty_dirs}")
    print(f"metadata saved: {metadata_path}")

    print("noise reuse summary:")
    for noise_run in noise_runs:
        key = str(noise_run.run_dir)
        print(f"  {noise_run.run_dir.name}: {len(used_intervals_by_noise[key])}")


if __name__ == "__main__":
    main()