#!/usr/bin/env python3
import argparse
from pathlib import Path
import soundfile as sf


def seconds_to_hms(sec: float):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./noisy")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    wav_files = sorted(root.rglob("*.wav"))

    total_sec = 0.0
    valid_count = 0

    for wav_path in wav_files:
        try:
            info = sf.info(str(wav_path))
            total_sec += info.frames / info.samplerate
            valid_count += 1
        except Exception as e:
            print(f"[SKIP] {wav_path} | {e}")

    print(f"wav files: {valid_count}")
    print(f"total seconds: {total_sec:.2f}")
    print(f"total minutes: {total_sec / 60:.2f}")
    print(f"total hours: {total_sec / 3600:.4f}")
    print(f"total hms: {seconds_to_hms(total_sec)}")


if __name__ == "__main__":
    main()