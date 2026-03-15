import cv2
import time
import datetime
import os
import numpy as np

from config.settings import settings
from core import holistic_detector, pose_classifier


class DataCollector:
    def __init__(
        self, data_dir="training_data", history_dir="history", latest_dir="latest"
    ):
        self.holistic_detector = holistic_detector.HolisticDetector()
        self.pose_classifier = pose_classifier.PoseClassifier()
        self.auto_collect = False
        self.auto_collect_capture_cooldown = 0.5
        self.current_label = 0
        self.frames_per_label = 100

        self.labels = {
            0: "knight_cheering",
            1: "hog_twerking",
            2: "princess_yawning",
            3: "crying_goblin",
            4: "king_laughing",
            5: "no_emote",
        }

        self.history_dir = history_dir
        self.latest_dir = latest_dir

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, self.history_dir), exist_ok=True)
        os.makedirs(os.path.join(data_dir, self.latest_dir), exist_ok=True)

        self.data_dir = data_dir

        self.empty_frames = 0
        self.max_empty_frames = 10

        self.auto_prepare_time = 5

    def _move_label_up(self):
        self.current_label += 1
        if self.current_label > len(self.labels) - 1:
            self.current_label = 0

    def _move_label_down(self):
        self.current_label -= 1
        if self.current_label < 0:
            self.current_label = len(self.labels) - 1

    def collect_data(self):
        collected_data = []

        cap = cv2.VideoCapture(0)

        KEY_LEFT = 2
        KEY_RIGHT = 3

        auto_collect_timer_current_time = time.time()
        auto_collect_timer_last_time = time.time()

        auto_started_time = 0

        saved_frames_current_label = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if ret is None or frame is None:
                self.empty_frames += 1
                if self.empty_frames >= self.max_empty_frames:
                    break
                else:
                    continue
            self.empty_frames = 0

            window_width = frame.shape[1]
            window_height = frame.shape[0]

            # collect data
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            holistic_landmarks = self.holistic_detector.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.holistic_detector.draw_landmarks(frame, holistic_landmarks)
            frame = cv2.flip(frame, 1)

            # handle auto collect logic
            is_preparing = time.time() < auto_started_time + self.auto_prepare_time

            if self.auto_collect and not is_preparing:
                auto_collect_timer_current_time = time.time()

                if (
                    auto_collect_timer_current_time - auto_collect_timer_last_time
                    > self.auto_collect_capture_cooldown
                ):
                    features = self.pose_classifier.compute_features(holistic_landmarks)
                    collected_data.append(features)
                    # Save data by auto here
                    if saved_frames_current_label >= self.frames_per_label:
                        self.save_data(self.labels[self.current_label], collected_data)
                        self.auto_collect = False
                        cur_label = self.current_label
                        self._move_label_up()
                        saved_frames_current_label = 0
                        collected_data.clear()
                        print(
                            f"Finished capturing data for '{self.labels[cur_label]}'. Next label is '{self.labels[self.current_label]}'. Stopping auto-collect."
                        )
                    else:
                        auto_collect_timer_last_time = auto_collect_timer_current_time
                        saved_frames_current_label += 1
                        print(f"{saved_frames_current_label} frames saved.")

            cv2.putText(frame, "(q) quit, (a) auto toggle", (10, 38), 2, 1, (255, 0, 0))
            cv2.putText(
                frame,
                f"Selected label: {self.labels[self.current_label]}",
                (10, 38 + 50),
                2,
                1,
                (255, 0, 0),
            )
            if is_preparing:
                cv2.putText(
                    frame,
                    f"{round((auto_started_time+self.auto_prepare_time)-time.time(), 1)}",
                    (window_width - 70, 40),
                    2,
                    1,
                    (0, 0, 255),
                )

            if self.auto_collect and not is_preparing:
                cv2.putText(
                    frame,
                    f"{saved_frames_current_label}",
                    (window_width - 70, 40),
                    4,
                    1,
                    (255, 0, 0),
                )

            cv2.imshow("frame", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                quit()

            if key == ord("a"):

                self.auto_collect = not self.auto_collect
                if self.auto_collect:
                    auto_started_time = time.time()
                    auto_collect_timer_current_time = time.time()
                    auto_collect_timer_last_time = time.time()
                print(f"Autocollect is {"on" if self.auto_collect else "off"}.")

            if not self.auto_collect:
                if key == KEY_LEFT:
                    self._move_label_down()
                    print(self.labels[self.current_label])
                if key == KEY_RIGHT:
                    self._move_label_up()
                    print(self.labels[self.current_label])

        print("Shutting down peacefully.")
        cap.release()
        cv2.destroyAllWindows()
        self.holistic_detector.close()

    def save_data(self, label, features):
        if isinstance(features, (list, tuple)):
            features = np.array(features)

        print(features)

        timestamp = datetime.datetime.now().strftime("%d-%m-%YT%H:%M:%S")

        history_file_name = f"{timestamp}.npy"
        latest_file_name = f"{label}_latest.npy"

        os.makedirs(os.path.join(self.data_dir, self.history_dir, label), exist_ok=True)
        np.save(
            os.path.join(self.data_dir, self.history_dir, label, history_file_name),
            features,
        )
        np.save(
            os.path.join(self.data_dir, self.latest_dir, latest_file_name), features
        )

    def load_data(self, path):
        return np.load(path)


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()
