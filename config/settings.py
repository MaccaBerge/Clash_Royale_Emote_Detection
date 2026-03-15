from mediapipe.python.solutions import drawing_utils
import pygame


class Screen:
    size = (
        int(pygame.display.Info().current_w * 0.9),
        int(pygame.display.Info().current_h * 0.9),
    )  # (1280, 720)
    fps = 30


class HolisticDrawingStyle:
    face_landmark_spec = drawing_utils.DrawingSpec((64, 224, 208), 1, 1)
    face_connection_spec = drawing_utils.DrawingSpec((64, 224, 208), 1, 1)

    right_hand_landmark_spec = drawing_utils.DrawingSpec((64, 224, 208), 1, 1)
    right_hand_connection_spec = drawing_utils.DrawingSpec((64, 224, 208), 1, 1)

    left_hand_landmark_spec = drawing_utils.DrawingSpec((64, 224, 208), 1, 1)
    left_hand_connection_spec = drawing_utils.DrawingSpec((64, 224, 208), 1, 1)

    pose_landmark_spec = drawing_utils.DrawingSpec((64, 224, 208), 1, 1)
    pose_connection_spec = drawing_utils.DrawingSpec((64, 224, 208), 1, 1)


class Style:
    holistic = HolisticDrawingStyle()


class General:
    small_image_random_speed_range = (300, 800)
    small_image_width = Screen.size[0] * 0.09


class Settings:
    screen = Screen()
    style = Style()
    general = General()


settings = Settings()
