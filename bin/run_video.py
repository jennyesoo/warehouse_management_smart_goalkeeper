
import argparse
from config import AppConfig
from Video_Surveillance import IntrusionMonitor

def parse_args() -> AppConfig:
    ap = argparse.ArgumentParser(description="Direction-violation detector with YOLOv4 + SORT")
    ap.add_argument("--video", help="Path to the video file. If omitted, webcam is used.")
    ap.add_argument("--min_area", type=int, default=200, help="Minimum contour area to consider as motion")
    ap.add_argument("--save_path", help="Directory to save alert frames (optional)")
    ap.add_argument("--direction", type=str, default="right", choices=["right", "left", "up", "down"],
                    help="Forbidden moving direction to detect")
    ap.add_argument("--limit_line_rate", type=int, default=2,
                    help="The ratio of frame width used to set the limit line position")
    ap.add_argument("--frame_size", type=int, default=480,
                    help="Resize input frames to a square of this width (pixels)")
    ap.add_argument("--freq", type=int, default=5,
                    help="Process one frame every N frames (throttling)")
    ap.add_argument("--computer_no", type=int, default=1, help="Device/computer identifier to embed in filename")
    args = ap.parse_args()

    return AppConfig(
        video=args.video,
        min_area=args.min_area,
        save_path=args.save_path,
        direction=args.direction,
        limit_line_rate=args.limit_line_rate,
        frame_size=args.frame_size,
        freq=args.freq,
        computer_no=args.computer_no,
    )


def main() -> None:
    cfg = parse_args()
    monitor = IntrusionMonitor(cfg)
    monitor.run()


if __name__ == "__main__":
    main()
