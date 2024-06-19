import concurrent.futures
import datetime
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model  # type: ignore


class Model:
    def __init__(self, model_path: str):
        self.ACTIONS = np.array([
            "_", "hello", "what's up", "how",
            "thanks", "you", "morning", "afternoon",
            "night", "me", "name", "fine",
            "happy", "yes", "no", "repeat",
            "please", "want", "good bye", "learn",
        ])

        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.LandmarkList = landmark_pb2.NormalizedLandmarkList
        self.NormalizedLandmark = landmark_pb2.NormalizedLandmark

        self.hand_task_path = "./tasks/hand_landmarker.task"
        self.pose_task_path = "./tasks/pose_landmarker.task"

        self.empty_hand_landmark = np.zeros((2, 21, 3))
        self.empty_pose_landmark = np.zeros(33 * 3)

        self.batch_size = 60
        self.threshold = 0.99

        self.model = load_model(model_path)

        self.init_task_vision()

    def init_task_vision(self):
        hand_base_options = python.BaseOptions(model_asset_path=self.hand_task_path)
        pose_base_options = python.BaseOptions(model_asset_path=self.pose_task_path)

        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.1,
            running_mode=self.VisionRunningMode.IMAGE,
        )

        # options for pose detection
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.1,
            running_mode=self.VisionRunningMode.IMAGE,
        )

        # create detectors
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    def extract_landmark(
        self,
        hand_results: vision.HandLandmarkerResult,
        pose_results: vision.PoseLandmarkerResult,
    ):
        pose_landmark = self.empty_pose_landmark
        hand_landmark = self.empty_hand_landmark

        if pose_results.pose_world_landmarks:
            pose_landmark = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_results.pose_world_landmarks[0]]
            ).flatten()

        # if no hand results are available, return the empty hand keypoints
        # and concatenate it with face and pose keypoints
        if not hand_results:
            return np.concatenate([pose_landmark, hand_landmark.flatten()])

        # iterate over the detected hand landmarks
        for index, hlm in enumerate(hand_results.hand_world_landmarks):
            # determine the hand index (0 for right hand, 1 for left hand) using handedness information
            handedness = hand_results.handedness[index][0].index

            # extract the keypoints for the current hand and assign them to the appropriate index
            hand_landmark[handedness] = np.array([[lm.x, lm.y, lm.z] for lm in hlm])

        return np.concatenate([pose_landmark, hand_landmark.flatten()])

    def process_frame(self, frame: int, image: np.ndarray):
        image = np.fliplr(image)

        # convert into mediapipe numpy type support uint8, uint16, or float32
        image = image.astype(np.uint8)

        # convert cv image to mediapipe image format before being passed to detectors
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        hand_results = self.hand_detector.detect(image=mp_image)
        pose_results = self.pose_detector.detect(image=mp_image)

        landmarks = self.extract_landmark(hand_results, pose_results)

        return frame, landmarks

    @staticmethod
    def format_timestamp(sec: float) -> str:
        # calculate hours, minutes, and seconds
        td = datetime.timedelta(seconds=sec)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # calculate milliseconds from the fractional part of seconds
        milliseconds = int((sec - int(sec)) * 1000)

        return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"

    def process_video(self, video: str):
        clip = VideoFileClip(video)

        vfps = clip.fps

        # store all predicted sign acted
        predictions = []

        # store all predicted senteces
        sentences = []

        # hold all frame, landmark from images that already being
        # detected by the mediapipe vision
        frames = []

        # store the x-length landmarks to be predict by the model
        sequences = []

        # banned word
        skip_word = "_"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_frame = {
                executor.submit(
                    self.process_frame,
                    frame,
                    image,
                ): frame
                for frame, image in enumerate(clip.iter_frames(fps=clip.fps))
            }

            for future in concurrent.futures.as_completed(future_to_frame):
                frame, landmarks = future.result()

                frames.append((frame, landmarks))

        clip.close()

        # sort the results by frame number
        # to ensure the order of the frame is correct
        frames.sort(key=lambda x: x[0])

        for frame, landmarks in frames:
            sequences.append(landmarks)

            if len(sequences) < self.batch_size:
                continue

            # ensure correct input shape by adding an extra dimension for batch size
            batch_motion = np.expand_dims(
                np.stack(sequences[-self.batch_size :]), axis=0
            )

            # predict the motion
            result = self.model.predict(batch_motion, verbose=0)[0]

            # get the predicted class and its confidence
            predicted = np.argmax(result)
            confidence = result[predicted]

            # append to the predictions and accuracies list
            predictions.append(predicted)

            # only keep the last 20 predictions and their accuracies
            predictions = predictions[-20:]

            predicted_sentence = self.ACTIONS[predicted]

            # determine most frequent prediction
            most_frequent_prediction = np.bincount(predictions[-10:]).argmax()

            if most_frequent_prediction != predicted:
                continue

            elif confidence < self.threshold:
                continue

            elif predicted_sentence == skip_word:
                continue

            elif not sentences or predicted_sentence != sentences[-1]["text"]:
                current_time_seconds = frame / vfps
                current_time = self.format_timestamp(current_time_seconds)

                sentences.append({
                    "text": self.ACTIONS[predicted],
                    "timestamp": current_time,
                })

        return sentences
