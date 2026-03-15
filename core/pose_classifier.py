import os
import joblib  # type: ignore
import pathlib
import numpy as np
from numpy.typing import ArrayLike
from typing import NamedTuple, Union, Tuple
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score, classification_report  # type: ignore

from core.typing import Landmark3D
from config import constants


class PoseClassifier:
    """Handles feature extraction, training and classification of human body poses.

    This class uses a `RandomForestClassifier` trained on pose features derived from
    MediaPipe Holistic landmark data. Each pose is represented as a 13-element NumPy
    array, where each element corresponds to a computed angle between three connected
    landmarks. These angle-bases features capture joint and facial geometry that can
    be used to predict the pose class of the most prominent person detected by
    MediaPipe.

    The class also provides utilities for saving and loading different models.

    Attributes:
        model: The RandomForestClassifier responsible for predicting human poses.
            Set to None by default.
        root_project_path: The path to the root of the project.
    """

    def __init__(self):
        self.model: Union[RandomForestClassifier, None] = None

        self.root_project_path: pathlib.Path = pathlib.Path(__file__).parent.parent

    def _calculate_angle(self, A: Landmark3D, B: Landmark3D, C: Landmark3D) -> float:
        """Calculates the angle between three 3D points.

        This method calculates the vectors BA and BC, and uses the dot-product to
        find the angle between them.

        Args:
            A: First 3D point.
            B: Second 3D point, vertext of the angle.
            C: Third 3D point.

        Returns:
            A float value representing the angle in degrees.
        """
        a = np.array([A.x, A.y, A.z])
        b = np.array([B.x, B.y, B.z])
        c = np.array([C.x, C.y, C.z])

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0

        ba_norm = ba / norm_ba
        bc_norm = bc / norm_bc

        return np.degrees(np.arccos(np.clip(np.dot(ba_norm, bc_norm), -1, 1)))

    def compute_features(self, holistic_pose_data: NamedTuple):
        """Computes angle-based features from MediaPipe Holistic landmark data.

        By extracting groups of three connected 3D points, the angles are calculated
        by finding the angle between these points. This improves model robustness by
        replacing position-dependent coordinates with relative angle-based features.

        Args:
            holistic_pose_data: MediaPipe Holistic landmark data.

        Returns:
            A 1D numpy array with 13 computed angle-based elements.

        Raises:
            AttributeError: If the provided Mediapipe result does not contain
                expected landmark attributes.
            IndexError: If required landmark indices are missing in the MediaPipe
                landmark list.
            TypeError: If `holistic_pose_data` is not a valid MediaPipe result
                object.
        """
        features = np.zeros(constants.Constants.general.prediction_model_num_features)

        if holistic_pose_data is None:
            return features

        pose_landmarks = holistic_pose_data.pose_landmarks  # type: ignore
        right_hand_landmarks = holistic_pose_data.right_hand_landmarks  # type: ignore
        left_hand_landmarks = holistic_pose_data.left_hand_landmarks  # type: ignore
        face_landmarks = holistic_pose_data.face_landmarks  # type: ignore

        if pose_landmarks:
            # ? Right elbow angle
            right_wrist = pose_landmarks.landmark[16]
            right_elbow = pose_landmarks.landmark[14]
            right_shoulder = pose_landmarks.landmark[12]
            right_elbow_angle = self._calculate_angle(
                right_wrist, right_elbow, right_shoulder
            )
            features[0] = right_elbow_angle

            # ? Left elbow angle
            left_wrist = pose_landmarks.landmark[15]
            left_elbow = pose_landmarks.landmark[13]
            left_shoulder = pose_landmarks.landmark[11]
            left_elbow_angle = self._calculate_angle(
                left_wrist, left_elbow, left_shoulder
            )
            features[1] = left_elbow_angle

            # ? Right shoulder angle
            right_hip = pose_landmarks.landmark[24]
            right_shoulder_angle = self._calculate_angle(
                right_elbow, right_shoulder, right_hip
            )
            features[2] = right_shoulder_angle

            # ? Left shoulder angle
            left_hip = pose_landmarks.landmark[23]
            left_shoulder_angle = self._calculate_angle(
                left_elbow, left_shoulder, left_hip
            )
            features[3] = left_shoulder_angle

            # ? Right mouth angle
            right_mouth = pose_landmarks.landmark[10]
            left_mouth = pose_landmarks.landmark[10]
            right_mouth_angle = self._calculate_angle(
                right_shoulder, right_mouth, left_mouth
            )
            features[6] = right_mouth_angle

        if right_hand_landmarks:
            # ? Right index finger pip angle
            right_index_finger_mcp = right_hand_landmarks.landmark[5]
            right_index_finger_pip = right_hand_landmarks.landmark[6]
            right_index_finger_tip = right_hand_landmarks.landmark[8]
            right_index_finger_pip_angle = self._calculate_angle(
                right_index_finger_mcp, right_index_finger_pip, right_index_finger_tip
            )
            features[7] = right_index_finger_pip_angle

            # ? Right pinky finger pip angle
            right_pinky_finger_mcp = right_hand_landmarks.landmark[17]
            right_pinky_finger_pip = right_hand_landmarks.landmark[18]
            right_pinky_finger_tip = right_hand_landmarks.landmark[20]
            right_pinky_finger_pip_angle = self._calculate_angle(
                right_pinky_finger_mcp, right_pinky_finger_pip, right_pinky_finger_tip
            )
            features[8] = right_pinky_finger_pip_angle

        if left_hand_landmarks:
            # ? Right index finger pip angle
            left_index_finger_mcp = left_hand_landmarks.landmark[5]
            left_index_finger_pip = left_hand_landmarks.landmark[6]
            left_index_finger_tip = left_hand_landmarks.landmark[8]
            left_index_finger_pip_angle = self._calculate_angle(
                left_index_finger_mcp, left_index_finger_pip, left_index_finger_tip
            )
            features[9] = left_index_finger_pip_angle

            # ? Right pinky finger pip angle
            left_pinky_finger_mcp = left_hand_landmarks.landmark[17]
            left_pinky_finger_pip = left_hand_landmarks.landmark[18]
            left_pinky_finger_tip = left_hand_landmarks.landmark[20]
            left_pinky_finger_pip_angle = self._calculate_angle(
                left_pinky_finger_mcp, left_pinky_finger_pip, left_pinky_finger_tip
            )
            features[10] = left_pinky_finger_pip_angle

        if face_landmarks:
            # ? Right center lip angle
            bottom_center_lip = face_landmarks.landmark[14]
            right_center_lip = face_landmarks.landmark[62]
            top_center_lip = face_landmarks.landmark[13]
            features[11] = self._calculate_angle(
                bottom_center_lip, right_center_lip, top_center_lip
            )

            # ? Left center lip angle
            left_center_lip = face_landmarks.landmark[291]
            features[12] = self._calculate_angle(
                bottom_center_lip, left_center_lip, top_center_lip
            )

        return features

    def train(
        self,
        X: Union[np.ndarray, ArrayLike],
        y: Union[np.ndarray, ArrayLike],
        save_model: bool = True,
        save_dir: Union[pathlib.Path, str] = "models",
    ) -> Tuple[float, str]:
        """Train the human pose prediction model.

        Args:
            X: A 2D numpy array of shape (n_samples, m_features) containing the
                training features.
            y: A 1D numpy array of shape (n_samples,) containing the class labels
                corresponding to each row in X.
            save_model: Decides if the program saves the model. If set to True
                the model is saved to the directory given by `save_dir`.
            save_dir: Path to directory to save model in, relative to the project
                root.

        Returns:
            Tuple[float,str]: A tuple containing:
            - float: The accuracy score of the model on the test set
                (a value between 0.0 and 1.0)
            - str: A classification report detailing precision, recall,
                and F1-score for each class.
        Raises:
            ValueError: If X or y has incompatible shapes or invalid data, or if
                scikit-learn rejects the the input during training or prediciton.
            TypeError: If X or y is not valid array-like objects.
            FileNotFoundError: If save_model is set to True, and the path does not
                exist.
            OSError: If saving the model fails due to file system issues.
            RuntimeError: If the RandomForestClassifier encounters an internal
                error.
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        report = str(classification_report(y_test, y_pred))

        if save_model:
            self.save_model(save_dir=save_dir)

        return accuracy, report

    def predict(self, X: Union[np.ndarray, ArrayLike]) -> np.ndarray:
        """Model predicts the human pose based on an array of features.

        Args:
            X: A 2D numpy array of shape (n_samples, m_features) containing the
                features used to predict the human pose.

        Returns:
            A 1D numpy array containing the predicted human pose.

        Raises:
            ValueError: If the model has not been laoded to memory.
            RuntimeError: If the model encounters an internal error.
        """
        if self.model is None:
            raise ValueError("The model has not been loaded to memory.")

        return self.model.predict(X)

    def save_model(self, save_dir: Union[pathlib.Path, str] = "models"):
        """Save model to file.

        Args:
            save_dir: Directory to save the model to, relative to the root
                of the project.

        Returns:
            None: Saves model to file and returns no data.

        Raises:
            ValueError: If the model is not loaded to memory.
            FileNotFoundError: If the directory does not exist.
            OSError: If saving the model fails due to a file system error.
            RuntimeError: If joblib encounters an internal error.
        """
        if self.model is None:
            raise ValueError("The model has not been loaded to memory.")

        full_dir_path = os.path.join(self.root_project_path, save_dir)
        os.makedirs(full_dir_path, exist_ok=True)

        joblib.dump(self.model, os.path.join(full_dir_path, "model.joblib"))

    def load_model(self, model_path: Union[pathlib.Path, str]):
        """Load model from file.

        Args:
            model_path: Path to the model file, relative to the project root.

        Returns:
            None: The model is loaded into the instance attribute `self.model`.

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError: If joblib encounters an internal error.
            OSError: If loading the model fails due to a file system error.
        """
        full_path = os.path.join(self.root_project_path, model_path)
        self.model = joblib.load(full_path)
