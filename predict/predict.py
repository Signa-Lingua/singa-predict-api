import cv2  # type: ignore
import os
import time
import numpy as np  # type: ignore

import mediapipe as mp  # type: ignore

from matplotlib import pyplot as plt # type: ignore
from mediapipe.tasks import python  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore
from mediapipe.framework.formats import landmark_pb2 # type: ignore


from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import BatchNormalization, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Reshape, Bidirectional  # type: ignore
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from datetime import datetime

class LoadModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ACTIONS = np.array(
            [
                "hello",
                "thanks",
                "i-love-you",
                "see-you-later",
                "I",
                "Father",
                "Mother",
                "Yes",
                "No",
                "Help",
                "Please",
                "Want",
                "What",
                "Again",
                "Eat",
                "Milk",
                "More",
                "Go To",
                "Bathroom",
                "Fine",
                "Like",
                "Learn",
                "Sign",
                "Done",
            ]
        )[:6]


        self.colors = [
            (245, 117, 16),
            (117, 245, 16),
            (16, 117, 245),
            (117, 117, 16),
            (16, 245, 117),
            (245, 117, 245)
        ]

        self.input_shape = (30, 1692)

        self.drawer = mp.solutions.drawing_utils # drawing utilities
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.LandmarkList = landmark_pb2.NormalizedLandmarkList
        self.NormalizedLandmark = landmark_pb2.NormalizedLandmark

        self.sequence_length = 30
        self.threshold = 0.5

        # Freeze all layers except the last few
        # for layer in model.layers[:-6]:
        #     layer.trainable = False

        self.init_model()
        self.init_detector()


    def init_model(self):
        model = Sequential()

        # data normalization
        model.add(BatchNormalization(input_shape=self.input_shape))

        # first Conv1D layer with L2 regularization
        model.add(
            Conv1D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=l2(0.01))
        )  # changed kernel size and filters
        model.add(MaxPooling1D(pool_size=2))

        # second Conv1D layer with L2 regularization
        model.add(
            Conv1D(filters=128, kernel_size=3, activation="relu", kernel_regularizer=l2(0.01))
        )  # changed kernel size and filters
        model.add(MaxPooling1D(pool_size=2))

        # third Conv1D layer with L2 regularization
        model.add(
            Conv1D(filters=256, kernel_size=3, activation="relu", kernel_regularizer=l2(0.01))
        )  # changed kernel size and filters
        model.add(MaxPooling1D(pool_size=2))

        # dense layer for feature extraction with L2 regularization
        model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))

        # bidirectional LSTM layer with L2 regularization
        model.add(
            Bidirectional(
                LSTM(
                    512, return_sequences=False, activation="relu", kernel_regularizer=l2(0.01)
                )
            )
        )

        # dense layers for classification with dropout for regularization
        model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))  # slightly higher dropout rate, so it's not overfitting
        model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))  # slightly higher dropout rate, so it's not overfitting

        model.add(Dense(self.ACTIONS.shape[0], activation="softmax"))

        # Load pre-trained weights
        # model.load_weights(rf"models/keras/asl-action-cnn-lstm_1l-6a-es_p30__rlr_f05_p10_lr1e5-2.9M.keras")
        model.load_weights(self.model_path)

        self.model = model


    def init_detector(self):
        face_base_options = python.BaseOptions(model_asset_path="./tasks/face_landmarker.task")
        hand_base_options = python.BaseOptions(model_asset_path="./tasks/hand_landmarker.task")
        pose_base_options = python.BaseOptions(model_asset_path="./tasks/pose_landmarker.task")

        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=self.VisionRunningMode.VIDEO,
        )

        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2,
            running_mode=self.VisionRunningMode.VIDEO,
        )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            output_segmentation_masks=True,
            running_mode=self.VisionRunningMode.VIDEO,
        )

        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        

    def extract_keypoints(self, face_results, pose_results, hand_results):
        """Extracts keypoints from face, pose, and hand results for dataset creation.

        Handles cases with zero, one, or two hands, assigning hand keypoints based
        on handedness information.

        Args:
        face_results: Object containing face landmark data (if available), assumed to
                        have a `face_landmarks` attribute with landmark data.
        pose_results: Object containing pose landmark data (if available), assumed to
                        have a `pose_landmarks` attribute with landmark data.
        hand_results: Object containing hand landmark data (if available), assumed to
                        have `hand_landmarks` and `handedness` attributes.

        Returns:
        A tuple containing three NumPy arrays representing flattened keypoints for face,
        pose, and hand, respectively. Empty arrays are used for missing modalities.
        """

        # extract face keypoints if available, otherwise return a zero-filled array
        face_keypoints = (
            np.array(
                [
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in face_results.face_landmarks[0]
                ]
            ).flatten()
            if face_results.face_landmarks
            else np.zeros(478 * 3)  # 478 landmarks with 3 coordinates each (x, y, z)
        )

        # extract pose keypoints if available, otherwise return a zero-filled array
        pose_keypoints = (
            np.array(
                [
                    [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in pose_results.pose_landmarks[0]
                ]
            ).flatten()
            if pose_results.pose_landmarks
            else np.zeros(33 * 4)  # 33 landmarks with 4 values each (x, y, z, visibility)
        )

        # initialize hand keypoints with zeros for two hands (right and left),
        # each with 21 landmarks and 3 coordinates
        hand_keypoints = np.zeros((2, 21, 3))

        # if no hand results are available, return the empty hand keypoints
        # and concatenate it with face and pose keypoints
        if not hand_results:
            return np.concatenate(
                [face_keypoints, pose_keypoints, hand_keypoints.flatten()]
            )

        # iterate over the detected hand landmarks
        for idx in range(len(hand_results.hand_landmarks)):
            # determine the hand index (0 for right hand, 1 for left hand) using handedness information
            handedness = hand_results.handedness[idx][0].index

            # extract the keypoints for the current hand and assign them to the appropriate index
            hand_keypoints[handedness] = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_results.hand_landmarks[idx]]
            )

        # flatten the hand keypoints array and concatenate it with face and pose keypoints
        return np.concatenate([face_keypoints, pose_keypoints, hand_keypoints.flatten()])
    

    def create_landmark_list(self, landmarks, num_keypoints):
        """Creates a LandmarkList protocol buffer from a list of landmarks or fills with empty values if no landmarks are provided.

        Args:
            landmarks: A list of landmark objects, each containing x, y, z coordinates.
            num_keypoints: The number of keypoints to be included in the LandmarkList.

        Returns:
            A LandmarkList containing the converted landmarks or empty values if no landmarks are provided.
        """
        # generate empty landmarks with all coordinates set to 0.0
        empty_landmarks = [
            self.NormalizedLandmark(x=0.0, y=0.0, z=0.0) for _ in range(num_keypoints)
        ]

        return self.LandmarkList(
            landmark=(
                # convert provided landmarks to NormalizedLandmark objects or use empty landmarks
                [self.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks]
                if landmarks
                else empty_landmarks
            )
        )
    

    def extract_keypoints_for_drawing(self, face_results, pose_results, hand_results):
        """Converts face, pose, and hand landmarks to corresponding protocol buffer lists for drawing.

        Args:
            face_results: Object containing face landmark detection results.
            pose_results: Object containing pose landmark detection results.
            hand_results: Object containing hand landmark detection results.

        Returns:
            A tuple containing three LandmarkList messages: face_landmarks, pose_landmarks, and hand_landmarks.
        """
        # convert face landmarks to LandmarkList, using empty values if no landmarks are present
        face_landmarks_proto = self.create_landmark_list(
            face_results.face_landmarks[0] if face_results.face_landmarks else None, 478 * 3
        )

        # convert pose landmarks to LandmarkList, using empty values if no landmarks are present
        pose_landmarks_proto = self.create_landmark_list(
            pose_results.pose_landmarks[0] if pose_results.pose_landmarks else None, 33 * 4
        )

        # convert hand landmarks to LandmarkList, using empty values if no landmarks are present
        hand_landmarks_proto = [
            self.create_landmark_list(hand_landmarks, 21 * 3)
            for hand_landmarks in (
                hand_results.hand_landmarks
                if hand_results.hand_landmarks
                else [None, None]  # two hands
            )
        ]

        return face_landmarks_proto, pose_landmarks_proto, hand_landmarks_proto
    

    def draw_detection_landmark(self,
        image,
        face_landmarks_proto=None,
        pose_landmarks_proto=None,
        hand_landmarks_proto=None,
    ):
        
        # draw landmark face
        self.drawer.draw_landmarks(
            image,
            face_landmarks_proto,
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            self.drawer.DrawingSpec(color=(80, 60, 20), thickness=1, circle_radius=1),
            self.drawer.DrawingSpec(color=(80, 146, 241), thickness=1, circle_radius=1),
        )

        # draw landmark pose
        self.drawer.draw_landmarks(
            image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            self.drawer.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
            self.drawer.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

        # draw landmark for both hand (right, left)
        for idx in range(len(hand_landmarks_proto)):
            self.drawer.draw_landmarks(
                image,
                hand_landmarks_proto[idx],
                mp.solutions.hands.HAND_CONNECTIONS,
                self.drawer.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                self.drawer.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
            )


    def predict_v(self, path: str):
        sequences = []
        sequence = []

        sentence = []
        predictions = []

        cap = cv2.VideoCapture(path)

        timestamp_ms = 0
        previous_timestamp_ms = 0

        start_time = time.time()
        isQuit = False

        while True:
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                break

            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get the current timestamp in milliseconds
            current_time = datetime.now()
            timestamp_ms = int(current_time.timestamp() * 1000)

            # Ensure timestamps are monotonically increasing
            if timestamp_ms <= previous_timestamp_ms:
                print(
                    f"Timestamp error: {timestamp_ms} is not greater than {previous_timestamp_ms}"
                )
                continue  # Skip the current frame if the timestamp is not increasing

            previous_timestamp_ms = timestamp_ms

            # Convert cv image to mediapipe image format before being passed to detectors
            annotated_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            try:
                face_results = self.face_detector.detect_for_video(
                    image=annotated_image, timestamp_ms=timestamp_ms + 1
                )

                hand_results = self.hand_detector.detect_for_video(
                    image=annotated_image, timestamp_ms=timestamp_ms
                )

                pose_results = self.pose_detector.detect_for_video(
                    image=annotated_image, timestamp_ms=timestamp_ms
                )
            except ValueError as ex:
                print(ex)
                continue

            # Extract keypoints (implement your extract_keypoints function)
            keypoints = self.extract_keypoints(face_results, pose_results, hand_results)
            sequences.append(keypoints)
            sequence = sequences[-30:]

            face_proto, pose_proto, hand_proto = self.extract_keypoints_for_drawing(
                face_results, pose_results, hand_results
            )

            self.draw_detection_landmark(
                image_rgb,
                face_landmarks_proto=face_proto,
                pose_landmarks_proto=pose_proto,
                hand_landmarks_proto=hand_proto,
            )

            if len(sequence) == self.sequence_length:
                # predict the action label based on the sequence of keypoints
                result = self.model.predict(
                    np.expand_dims(
                        sequence, axis=0
                    )  # expanded to include a batch dimension before fed to the model
                )[0]

                # action class with the highest confidence score
                predictions.append(np.argmax(result))

                # NOTE: If the current prediction matches the most common prediction over the last 10 frames,
                #       it suggests that the current action is likely intentional and
                #       consistent with recent actions, rather than a momentary anomaly.
                if np.unique(predictions[-10:])[0] == np.argmax(result):

                    # check if the confidence score of the current prediction index is above the threshold.
                    if result[np.argmax(result)] > self.threshold:

                        # checks if there are any elements in the sentence list.
                        # If it's not empty, it means there are already recognized actions in the sentence.
                        if len(sentence) > 0:
                            # compares the current predicted action
                            if self.ACTIONS[np.argmax(result)] != sentence[-1]:
                                sentence.append(self.ACTIONS[np.argmax(result)])
                        else:
                            # no recognized actions yet
                            sentence.append(self.ACTIONS[np.argmax(result)])

            # cv2.imshow("MediaPipe Detection", cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        # cv2.destroyAllWindows()
        return sentence
