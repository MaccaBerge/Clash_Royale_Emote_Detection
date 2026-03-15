from typing import Union
import pygame
import cv2
import numpy as np
import random

from config import constants, settings
from core import (
    app,
    base_state,
    asset_manager,
    holistic_detector,
    pose_classifier,
    logger,
)
from components import timer, small_moving_image, timer


class DetectionState(base_state.BaseState):
    def __init__(self, app_object) -> None:
        self.app: app.App = app_object
        self.asset_manager: asset_manager.AssetManager = asset_manager.asset_manager
        self.holistic_detector: holistic_detector.HolisticDetector = (
            holistic_detector.HolisticDetector()
        )
        self.pose_classifier: pose_classifier.PoseClassifier = (
            pose_classifier.PoseClassifier()
        )
        self.pose_classifier.load_model(constants.Constants.path.prediction_model_path)
        self._cap: Union[cv2.VideoCapture, None] = None

        self.moving_images_manager = small_moving_image.SmallMovingImageManager()

        self.emote_surface = pygame.Surface(settings.settings.screen.size)
        self.video_surface = pygame.Surface(
            (
                settings.settings.screen.size[0] / 2,
                settings.settings.screen.size[1] / 2,
            )
        )

        self.current_emote = None
        self.current_emote_rect = None
        self.current_emote_image = None
        self.last_emote = None

        self.dual_display_mode = False

        self.sound_timer = timer.timer_manager.create_timer(1, self._play_sound)

    def _play_sound(self):
        if self.current_emote is None:
            return

        sound = asset_manager.asset_manager.get_sound(self.current_emote)
        if sound is None:
            logger.logger.warning(f"No sound for emote {self.current_emote}")
            return

        try:
            sound.play()
        except Exception:
            logger.logger.exception(f"The sound {sound} failed to play")

    def enter(self):
        self._cap = self.app.get_video_input()

    def exit(self):
        if self.holistic_detector is not None:
            self.holistic_detector.close()

    def update(self, dt: float, surface: pygame.Surface) -> str | None:
        self.current_emote = None
        self.emote_surface.fill((255, 255, 255))
        if self._cap is None:
            print("Show error screen.")
            return

        ret, frame = self._cap.read()

        #! Make a maxiumum amount of retries on empty frames and add error screen
        if ret is None or frame is None:
            return

        # turning off writeable gives a small performance boost (see google holistic documentation)
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_landmarks = self.holistic_detector.process(frame)
        frame.flags.writeable = True

        if holistic_landmarks.pose_landmarks is not None:  # type: ignore
            features = self.pose_classifier.compute_features(holistic_landmarks)
            features = features[np.newaxis, ...]  # features.reshape(1, -1)
            prediction = self.pose_classifier.predict(features)

            if isinstance(prediction, np.ndarray):
                prediction = str(prediction[0])
                self.current_emote = prediction

        if self.current_emote not in [None, "no_emote"]:

            self.moving_images_manager.create_image(
                asset_manager.asset_manager.get_image(f"{self.current_emote}_small"),
                (random.randint(60, settings.settings.screen.size[0] - 60), -80),
                random.randint(
                    *settings.settings.general.small_image_random_speed_range
                ),
                direction=[0, 1],
            )
            if self.current_emote != self.last_emote:
                self.sound_timer.reset()
                self._play_sound()

                self.current_emote_image = asset_manager.asset_manager.get_image(
                    f"{self.current_emote}_big"
                )
                self.current_emote_rect = self.current_emote_image.get_rect(
                    center=(
                        settings.settings.screen.size[0] / 2,
                        settings.settings.screen.size[1] / 2,
                    )
                )
            if (
                self.current_emote_image is not None
                and self.current_emote_rect is not None
            ):
                self.emote_surface.blit(
                    self.current_emote_image, self.current_emote_rect
                )

        self.moving_images_manager.update(dt, self.emote_surface)
        self.last_emote = self.current_emote

        if self.dual_display_mode:
            surface.fill((0, 0, 0))

            sized_frame = cv2.resize(frame, self.video_surface.size)
            self.holistic_detector.draw_landmarks(sized_frame, holistic_landmarks)
            sized_frame = cv2.flip(sized_frame, 1)
            frame_surface = pygame.image.frombuffer(
                sized_frame.tobytes(), sized_frame.shape[1::-1], "RGB"
            ).convert()
            self.video_surface.blit(frame_surface, (0, 0))

            emote_surf = pygame.transform.scale_by(
                self.emote_surface, ((surface.width / 2) / self.emote_surface.width)
            )
            surface.blit(
                self.video_surface,
                (0, (surface.height / 2) - (self.video_surface.height / 2)),
            )
            surface.blit(
                emote_surf,
                (surface.width / 2, (surface.height / 2) - (emote_surf.height / 2)),
            )
        else:
            surface.blit(self.emote_surface, (0, 0))

    def handle_event(self, event: pygame.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.dual_display_mode = not self.dual_display_mode
