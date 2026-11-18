"""Restore faces in a directory of MP4 videos."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path, help="Directory containing MP4 videos")
    args = parser.parse_args()
    video_dir = args.video_dir.expanduser().resolve()
    if not video_dir.is_dir():
        parser.error("video_dir must be an existing directory")

    from face_restore_gfpgan_thread_pipeline import main as restore_videos

    restore_videos(str(video_dir))


if __name__ == "__main__":
    main()
