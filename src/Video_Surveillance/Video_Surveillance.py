
import os
import cv2
import time
import logging
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from imutils.video import VideoStream

from config import AppConfig
from sort import Sort
from object_detection import load_network, object_detection
from util import (
    detect_direction,
    center_record,
    check_direction,
    draw_boxes,
    get_file_name,
    yolo2sort,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


class IntrusionMonitor:
    """
    封裝整條流程：
      - 影片/攝影機輸入
      - 背景減除 (KNN)
      - YOLO 偵測 + SORT 多目標追蹤
      - 違規方向檢查與告警影像輸出
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

        # I/O：相機或檔案
        self.vs = self._open_stream(cfg.video)

        # 背景分離器（KNN）
        self.back_sub = cv2.createBackgroundSubtractorKNN()

        # 物件偵測模型 (YOLOv4)
        self.network, self.class_names, self.class_colors = load_network()

        # 多目標追蹤器
        self.tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.05)

        # 違規方向的警戒線
        self.alarm_limit = detect_direction(
            cfg.direction, cfg.frame_size, cfg.limit_line_rate
        )

        # 追蹤狀態（重設門檻用）
        self.frame_count = 0
        self.iteration = 0

        # 在每一輪「活動期」使用的狀態容器
        self.pts: Dict[int, Deque[Tuple[int, int]]] = {}
        self.alarmed_ids: List[int] = []

        # 準備輸出資料夾
        if cfg.save_path:
            os.makedirs(cfg.save_path, exist_ok=True)

        # 統一影像尺寸：若需要將影像縮放為正方形（以 width=frame_size）
        self.resize_to_square = cfg.frame_size

        logging.info("IntrusionMonitor initialized.")

    def run(self) -> None:
        """
        以 freq 節流讀取與處理影格；空幀或 EOF 結束。
        """
        try:
            while True:
                should_process = (self.frame_count % max(self.cfg.freq, 1) == 0)
                frame = self._read_frame()
                if frame is None:
                    logging.info("No more frames. Exit.")
                    break

                if should_process:
                    frame = self._ensure_square(frame, self.resize_to_square)

                    # 背景減除，偵測是否存在「明顯運動」
                    motion_exists = self._has_significant_motion(frame, self.cfg.min_area)

                    if motion_exists:
                        # 做 YOLO 偵測
                        image_resized, detections = object_detection(
                            frame, self.network, self.class_names, self.class_colors
                        )

                        # 如果偵測到東西且這是一輪活動的開始，重置活動期容器
                        if len(detections) != 0 and self.iteration == 0:
                            self._start_activity_period()

                        # 若尚在活動期，更新追蹤與告警
                        if self._in_activity_period():
                            self._update_tracking_and_alert(image_resized, detections)

                        self.iteration += 1
                    else:
                        # 沒有明顯運動了，若之前在活動期，做收尾重置
                        if self._in_activity_period():
                            self._end_activity_period()

                self.frame_count += 1

        finally:
            self._release()

    def _open_stream(self, path: Optional[str]):
        """
        開啟串流或攝影機。
        - path is None: webcam (VideoStream)
        - else: file (cv2.VideoCapture)
        """
        if path is None:
            logging.info("Opening webcam (VideoStream).")
            vs = VideoStream(src=0).start()
            time.sleep(2.0)  # hot start
            return vs
        else:
            logging.info(f"Opening video file: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {path}")
            return cap

    def _read_frame(self) -> Optional[np.ndarray]:
        """
        統一取得影格：
        - VideoStream：vs.read() -> frame
        - VideoCapture：cap.read() -> (ok, frame)
        回傳 None 代表 EOF / 取幀失敗。
        """
        if isinstance(self.vs, VideoStream):
            frame = self.vs.read()
            return frame if frame is not None else None
        else:
            ok, frame = self.vs.read()
            return frame if ok else None

    def _release(self) -> None:
        """
        統一釋放資源。
        """
        try:
            if isinstance(self.vs, VideoStream):
                self.vs.stop()
            else:
                self.vs.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logging.info("Resources released.")


    def _ensure_square(self, frame: np.ndarray, width: int) -> np.ndarray:
        """
        可選：把 frame resize 成正方形（width x width）。
        若不想裁切，這裡使用等比縮放後再邊緣填補方式也可；目前採單純 resize。
        """
        if width <= 0:
            return frame
        return cv2.resize(frame, (width, width))

    def _has_significant_motion(self, frame: np.ndarray, min_area: int) -> bool:
        """
        背景減除 + 門檻化 + 形態學開運算+dilate，若任一輪廓面積 >= min_area 即視為有明顯運動。
        """
        fg_mask = self.back_sub.apply(frame)
        # KNN 的前景遮罩亮像素通常接近 255，門檻 244 取近白
        thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None)
        thresh = cv2.dilate(thresh, None, iterations=5)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in cnts]
        return any(a >= min_area for a in areas)


    def _start_activity_period(self) -> None:
        """
        一輪活動（偵測到物件且持續）剛開始：重置追蹤容器，避免長時執行無限制累積。
        """
        self.pts = {}  # id -> deque of centers
        self.alarmed_ids = []
        logging.debug("Start activity period.")

    def _in_activity_period(self) -> bool:
        return self.iteration >= 0  # 用 iteration >= 0 表示活動啟動過；你也可改更嚴謹條件

    def _end_activity_period(self) -> None:
        """
        一輪活動結束：歸零迭代計數、清理暫存。
        """
        self.iteration = 0
        self.pts.clear()
        self.alarmed_ids.clear()
        logging.debug("End activity period.")


    def _update_tracking_and_alert(
        self, image_resized: np.ndarray, detections: List
    ) -> None:
        """
        - 把 YOLO 偵測結果轉給 SORT
        - 更新追蹤邊框與 ID
        - 確認是否有 ID 跨越違規方向的警戒線
        - 新違規：畫框、存檔（避免重複觸發）
        """
        if len(detections) != 0:
            dets, items = yolo2sort(detections)
            dets = np.asarray(dets, dtype=float)
        else:
            dets = np.empty((0, 5))
            items = [[None, None]]

        # 更新多目標追蹤（x1, y1, x2, y2, track_id）
        track_bbs_ids = self.tracker.update(dets)

        # 更新每個 ID 的中心點軌跡
        self.pts = center_record(track_bbs_ids, self.pts)

        # 檢查是否跨越不允許方向的限制線
        to_alarm_dt, to_alarm_ids = check_direction(
            track_bbs_ids,
            self.cfg.direction,
            self.alarm_limit,
            self.pts
        )

        # 只對「新」違規 ID 觸發
        new_ids = list(set(to_alarm_ids) - set(self.alarmed_ids))
        if len(new_ids) > 0:
            analyzed = draw_boxes(to_alarm_dt, image_resized, self.class_colors, items)
            analyzed = cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB)

            # 準備檔名
            filename = get_file_name(self.cfg.computer_no)
            out_path = filename if not self.cfg.save_path else os.path.join(self.cfg.save_path, filename)

            try:
                cv2.imwrite(out_path, analyzed)
                logging.info(f"Alert frame saved: {out_path}")
            except Exception as e:
                logging.warning(f"Failed to save frame to {out_path}: {e}")

            # 更新已告警 ID，避免重複通知
            self.alarmed_ids.extend(new_ids)

