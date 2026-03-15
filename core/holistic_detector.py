from typing import NamedTuple
from mediapipe.python.solutions import holistic
from mediapipe.python.solutions import drawing_utils
from numpy import ndarray

from config.settings import settings


class HolisticDetector:
    """Class for landmark detection and rendering.

    This class uses the `holistic.Holistic` model from MediaPipe to predict
    the position of landmarks on a human body. This class also supports drawing
    these landmarks and connections to an image.
    """

    def __init__(self):
        self.holistic = holistic.Holistic()

    def process(self, frame: ndarray) -> NamedTuple:
        """Processes an RGB image and returns the pose landmarks, left and right
        hand landmarks, and face landmarks on the most prominent person detected.

        Args:
            image: An RGB image represented as a numpy ndarray.

        Raises:
            RuntimeError: If the underlying graph throws any error.
            ValueError: If the input image is not three channel RGB.
        """
        return self.holistic.process(frame)

    def draw_landmarks(self, frame: ndarray, holistic_pose_data: NamedTuple):
        """Draws the landmarks and connections on the image.

        Using MediaPipe's `drawing_utils.draw_landmarks` to draw the landmarks and
        connections of face_landmarks, right_hand_landmarks, left_hand_landmarks and
        pose_landmarks.

        Args:
            frame: A three channel BGR image represented as a ndarray.
            holistic_pose_data: MediaPipe Holistic landmark data.

        Raises:
            ValueError: If one of the following:
                a) If the input image is not three channel BGR.
                b) If any connections contain invalid landmark index.
        """

        face_landmarks = holistic_pose_data.face_landmarks  # type: ignore
        right_hand_landmarks = holistic_pose_data.right_hand_landmarks  # type: ignore
        left_hand_landmarks = holistic_pose_data.left_hand_landmarks  # type: ignore
        pose_landmarks = holistic_pose_data.pose_landmarks  # type: ignore

        if face_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                holistic.FACEMESH_TESSELATION,  # type: ignore
                landmark_drawing_spec=settings.style.holistic.face_landmark_spec,
                connection_drawing_spec=settings.style.holistic.face_connection_spec,
            )

        if right_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                right_hand_landmarks,
                holistic.HAND_CONNECTIONS,  # type: ignore
                landmark_drawing_spec=settings.style.holistic.right_hand_landmark_spec,
                connection_drawing_spec=settings.style.holistic.right_hand_connection_spec,
            )

        if left_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                left_hand_landmarks,
                holistic.HAND_CONNECTIONS,  # type: ignore
                landmark_drawing_spec=settings.style.holistic.left_hand_landmark_spec,
                connection_drawing_spec=settings.style.holistic.left_hand_connection_spec,
            )

        if pose_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                pose_landmarks,
                holistic.POSE_CONNECTIONS,  # type: ignore
                landmark_drawing_spec=settings.style.holistic.pose_landmark_spec,
                connection_drawing_spec=settings.style.holistic.pose_connection_spec,
            )

    def close(self):
        """Release all underlying MediaPipe resources."""
        self.holistic.close()
