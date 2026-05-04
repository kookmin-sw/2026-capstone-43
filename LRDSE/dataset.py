import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.audio.preprocess import (
    AudioPreprocessConfig,
    load_wav,
    preprocess_pair_for_train,
)

from src.condition.preprocess import (
    ConditionPreprocessConfig,
    preprocess_condition_for_train,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_manifest_path(path_str: str) -> str:
    """
    Resolve potentially stale absolute paths stored in manifest.csv.

    Typical stale prefix in this project:
        /.../robot_denoising/LRDSE/...
    Current root:
        .../robot_denoising/2026-capstone-43/LRDSE
    """
    if not path_str:
        return path_str

    path = Path(path_str)
    if path.exists():
        return str(path)

    # 1) Relative path in manifest -> interpret as project-root relative.
    if not path.is_absolute():
        candidate = (PROJECT_ROOT / path).resolve()
        if candidate.exists():
            return str(candidate)

    # 2) Stale absolute root -> rewrite by keeping tail after /robot_denoising/LRDSE/.
    normalized = str(path).replace("\\", "/")
    marker = "/robot_denoising/LRDSE/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = (PROJECT_ROOT / suffix).resolve()
        if candidate.exists():
            return str(candidate)

    return str(path)


def load_mono_audio(path, target_sr=16000):
    """
    train.py sample 저장에서 원본 wav를 다시 읽기 위한 compatibility 함수.
    return: wav [T], sr
    """
    cfg = AudioPreprocessConfig(sample_rate=target_sr)
    wav, sr = load_wav(str(path), cfg)
    return wav.squeeze(0).contiguous(), sr


def infer_run_dir(row):
    """
    condition 파일이 들어있는 run directory 추론.

    우선순위:
        1. manifest에 run_dir 컬럼이 있으면 사용
        2. noisy_wav의 parent directory 사용

    예상 구조:
        noisy/{speaker_id}/{book_id}/{source_id}/source.wav
        noisy/{speaker_id}/{book_id}/{source_id}/anchors.json
        noisy/{speaker_id}/{book_id}/{source_id}/lowstate_segment.jsonl
    """
    run_dir = row.get("run_dir", "")

    if run_dir:
        return str(Path(run_dir))

    noisy_wav = row.get("noisy_wav", "")

    if not noisy_wav:
        raise ValueError("Cannot infer run_dir because noisy_wav is empty")

    return str(Path(noisy_wav).parent)


class SpeechEnhancementDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        target_sr=16000,
        target_length=32640,
        n_fft=510,
        hop_length=128,
        win_length=510,
        center=True,
        num_frames=256,
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        normalize="noisy",
        random_crop=True,
        valid_only=True,
        limit=None,

        use_condition=True,
        raw_force_scale=220.0,
        d_force_scale=9220.325595510363,
        condition_smooth_win=1,
    ):
        self.manifest_path = Path(manifest_path)
        self.random_crop = random_crop
        self.use_condition = use_condition

        if win_length != n_fft:
            raise ValueError(
                f"Current preprocess.py uses win_length == n_fft. "
                f"Got win_length={win_length}, n_fft={n_fft}"
            )

        self.cfg = AudioPreprocessConfig(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            num_frames=num_frames,
            center=center,
            spec_factor=spec_factor,
            spec_abs_exponent=spec_abs_exponent,
            normalize=normalize,
        )

        self.cond_cfg = ConditionPreprocessConfig(
            raw_force_scale=raw_force_scale,
            d_force_scale=d_force_scale,
            smooth_win=condition_smooth_win,
        )

        if target_length != self.cfg.train_target_len:
            raise ValueError(
                f"target_length mismatch: got {target_length}, "
                f"expected {self.cfg.train_target_len} from "
                f"(num_frames - 1) * hop_length"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")

        rows = []
        path_remap_count = 0
        dropped_invalid_path_count = 0

        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if valid_only and str(row.get("valid", "0")) != "1":
                    continue

                noisy_wav = row.get("noisy_wav", "")
                clean_wav = row.get("clean_wav", "")

                if not noisy_wav or not clean_wav:
                    continue

                resolved_noisy = resolve_manifest_path(noisy_wav)
                resolved_clean = resolve_manifest_path(clean_wav)

                if resolved_noisy != noisy_wav or resolved_clean != clean_wav:
                    path_remap_count += 1
                    row = dict(row)
                    row["noisy_wav"] = resolved_noisy
                    row["clean_wav"] = resolved_clean

                if not Path(row["noisy_wav"]).exists() or not Path(row["clean_wav"]).exists():
                    dropped_invalid_path_count += 1
                    continue

                rows.append(row)

        if limit is not None:
            rows = rows[:limit]

        if len(rows) == 0:
            raise RuntimeError(f"no valid samples found in manifest: {self.manifest_path}")

        self.rows = rows

        if path_remap_count > 0:
            print(
                f"[dataset] remapped stale manifest paths: {path_remap_count} rows "
                f"(root={PROJECT_ROOT})"
            )
        if dropped_invalid_path_count > 0:
            print(
                f"[dataset] dropped rows with missing files: {dropped_invalid_path_count}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        out = preprocess_pair_for_train(
            clean_path=row["clean_wav"],
            noisy_path=row["noisy_wav"],
            cfg=self.cfg,
            random_crop=self.random_crop,
        )

        noisy_stft = out["noisy_2ch"].float()
        clean_stft = out["clean_2ch"].float()

        run_dir = infer_run_dir(row)

        sample = {
            "noisy_stft": noisy_stft,   # [2, 256, 256], transformed STFT
            "clean_stft": clean_stft,   # [2, 256, 256], transformed STFT
            "noisy_wav": out["noisy_wave"].squeeze(0).float(),
            "clean_wav": out["clean_wave"].squeeze(0).float(),
            "normfac": out["normfac"].float(),
            "meta": {
                "id": row.get("id", ""),
                "speaker_id": row.get("speaker_id", ""),
                "book_id": row.get("book_id", ""),
                "source_id": row.get("source_id", ""),
                "noisy_wav": row.get("noisy_wav", ""),
                "clean_wav": row.get("clean_wav", ""),
                "run_dir": run_dir,
                "start": out["start"],
            },
        }

        if self.use_condition:
            cond_out = preprocess_condition_for_train(
                run_dir=run_dir,
                crop_start_sample=int(out["start"].item()),
                num_frames=noisy_stft.shape[-1],
                hop_length=self.cfg.hop_length,
                freq_bins=noisy_stft.shape[-2],
                cfg=self.cond_cfg,
            )

            sample["cond"] = cond_out["cond_8ch"].float()          # [8, 1024]
            sample["cond_times"] = cond_out["cond_times"].float()  # [1024]
            sample["cond_mask"] = cond_out["cond_mask"].bool()     # [1024]
            sample["real_token_count"] = cond_out["real_token_count"]

        return sample
