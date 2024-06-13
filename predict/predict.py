import time

import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import matplotlib.pyplot as plt
import concurrent.futures
from moviepy.editor import VideoFileClip

class LoadModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ACTIONS = np.array(
            [
                "_", "hello", "thanks", "i-love-you", "I", "Yes", "No", "Help", "Please",
                "Want", "Eat", "More", "Bathroom", "Learn", "Sign",
            ]
        )[:8]


        self.colors = [
            (245, 117, 16),
            (117, 245, 16),
            (16, 117, 245),
            (117, 117, 16),
            (16, 245, 117),
            (245, 117, 245),
        ]

        self.input_shape = (30, 1692)

        self.drawer = mp.solutions.drawing_utils # drawing utilities
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.LandmarkList = landmark_pb2.NormalizedLandmarkList
        self.NormalizedLandmark = landmark_pb2.NormalizedLandmark

        self.empty_hand_landmark = np.zeros((2, 21, 3))  # right hand and left hand
        self.empty_pose_landmark = np.zeros(33 * 3)

        self.sequence_length = 30
        self.threshold = 0.5

        # Freeze all layers except the last few
        # for layer in model.layers[:-6]:
        #     layer.trainable = False

        self.init_model()
        self.init_detector()

        self.empty_pose_landmarks = self.to_landmark_list(
            [self.NormalizedLandmark(x=0.0, y=0.0, z=0.0) for _ in range(33 * 4)]
        )

        self.empty_hand_landmarks = self.to_landmark_list(
            [self.NormalizedLandmark(x=0.0, y=0.0, z=0.0) for _ in range(21 * 3)]
        )

    def init_model(self):
        try:
            model = tf.keras.models.load_model(self.model_path)
            self.model = model
        except Exception as e:
            print(f"{e}")

    def init_detector(self):
        hand_base_options = python.BaseOptions(model_asset_path="./tasks/hand_landmarker.task")
        pose_base_options = python.BaseOptions(model_asset_path="./tasks/pose_landmarker.task")

        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2,
            min_hand_detection_confidence=0.8,
            min_hand_presence_confidence=0.9,
            min_tracking_confidence=0.8,
            running_mode=self.VisionRunningMode.IMAGE,
        )

        # options for pose detection
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.95,
            min_pose_presence_confidence=0.95,
            min_tracking_confidence=0.95,
            running_mode=self.VisionRunningMode.IMAGE,
        )

        # create detectors
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        self.hand_detector = hand_detector
        self.pose_detector = pose_detector


    def to_landmark_data(self, hand_results: vision.HandLandmarkerResult, pose_results: vision.PoseLandmarkerResult):
        """
        Extract keypoints from pose and hand results for dataset creation.
        """
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


    def to_landmark_list(self, landmarks):
        """
        Create a LandmarkList from a list of landmarks or fill with empty values if no landmarks are provided.
        """
        return self.LandmarkList(
            landmark=([self.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks])
        )



    def to_drawing_landmark(self, hand_results, pose_results):
        """
        Convert pose and hand landmarks to LandmarkList for drawing.
        """
        pose_landmarks = (
            self.to_landmark_list(pose_results.pose_landmarks[0])
            if pose_results.pose_landmarks
            else self.empty_pose_landmarks
        )

        hand_landmarks = [self.empty_hand_landmarks, self.empty_hand_landmarks]

        if not hand_results:
            return pose_landmarks, None

        # iterate over the detected hand landmarks
        for index, hand_landmark in enumerate(hand_results.hand_landmarks):
            # determine the hand index (0 for right hand, 1 for left hand) using handedness information
            handedness = hand_results.handedness[index][0].index

            # extract the keypoints for the current hand and assign them to the appropriate index
            hand_landmarks[handedness] = self.to_landmark_list(hand_landmark)

        return pose_landmarks, hand_landmarks


    def draw_landmark(self, image, hand_landmarks, pose_landmarks):
        """
        Draw detected landmarks on the image.
        """
        self.drawer.draw_landmarks(
            image,
            pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            self.drawer.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
            self.drawer.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

        if not hand_landmarks:
            return

        for hand_landmarks in hand_landmarks:
            self.drawer.draw_landmarks(
                image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                self.drawer.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                self.drawer.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
            )




    def process_frame(self, frame, image, threshold, skip_word):
        start_time = time.time()

        # Convert into mediapipe numpy type support uint8, uint16, or float32
        image = image.astype(np.uint8)

        # Convert cv image to mediapipe image format before being passed to detectors
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        try:
            hand_results = self.hand_detector.detect(image=mp_image)
            pose_results = self.pose_detector.detect(image=mp_image)

            landmarks = self.to_landmark_data(hand_results, pose_results)
        except:
            print(f"frame {frame} skipped")
            return frame, None, time.time() - start_time

        return frame, landmarks, time.time() - start_time


    def predict_v(self, path: str):
        try:
            clip = VideoFileClip(path)
        except Exception as e:
            print(f"{e}")
        
        avg_exec_time = []

        predictions = []
        sequences = []

        sentence = [] 
        threshold = 0.2
        skip_word = "_"

        results = []
        batch_size = 60

        cap = cv2.VideoCapture(path)

        # frame_rate = cap.get(cv2.CAP_PROP_FPS)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # duration = frame_count / frame_rate  # Duration of the video in seconds

        timestamp_ms = 0
        previous_timestamp_ms = 0

        start_time = time.time()
        isQuit = False

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_frame = {
                executor.submit(
                    self.process_frame,
                    frame,
                    image,
                    threshold,
                    skip_word,
                ): frame
                for frame, image in enumerate(clip.iter_frames(fps=clip.fps))
            }

            for future in concurrent.futures.as_completed(future_to_frame):
                frame, landmarks, exec_time = future.result()
                avg_exec_time.append(exec_time)
            

                if landmarks is not None:
                    results.append((frame, landmarks))

        print(avg_exec_time)
        # sort the results by frame number to ensure the order is correct
        results.sort(key=lambda x: x[0])

        for _, landmarks in results:
            sequences.append(landmarks)

            if len(sequences) < batch_size:
                continue

            timestamp_ms = int((time.time() - start_time) * 1000)

            # collect a batch of sequences
            batch_motion = np.stack(sequences[-batch_size:])
            # sequences = sequences[
            #     -(batch_size - 50) :
            # ]  # keep the last 10 sequences for overlap
            sequences = []

            # ensure correct input shape by adding an extra dimension for batch size
            batch_motion = np.expand_dims(batch_motion, axis=0)

            # predict the entire batch
            batch_result = self.model.predict(batch_motion, verbose=0)

            print(batch_result)
            print("="*50)

            for result in batch_result:
                # len of results is 480 (which is the total frame)?
                predicted = np.argmax(result)

                if (not result[predicted] > threshold) or not (
                    self.ACTIONS[predicted] != skip_word
                ):
                    continue

                if not predictions or predicted != predictions[-1]:
                    elapsed_time = time.time() - start_time
                    hours, remainder = divmod(int(elapsed_time), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)
                    current_time = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


                    data = {
                        "text": self.ACTIONS[predicted],
                        "timestamp": current_time,
                    }
                    predictions.append(data)
    
        return predictions
        
        
        while True:
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                break

            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # # Get the current timestamp in milliseconds
            # current_time = datetime.now()
            # timestamp_ms = int(current_time.timestamp() * 1000)

            # Calculate timestamp based on frame rate
            timestamp_ms = int((time.time() - start_time) * 1000)
            

            # Ensure timestamp is within video duration
            if timestamp_ms > duration * 1000:
                break

            # Ensure timestamps are monotonically increasing
            if timestamp_ms <= previous_timestamp_ms:
                print(
                    f"Timestamp error: {timestamp_ms} is not greater than {previous_timestamp_ms}"
                )
                continue  # Skip the current frame if the timestamp is not increasing

            # print(f"Timestamp: {timestamp_ms} ms")

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

                    # current_time = time.strftime("%H:%M:%S.%f", time.gmtime(time.time() - start_time))

                    elapsed_time = time.time() - start_time
                    hours, remainder = divmod(int(elapsed_time), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)

                    current_time = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
                    ## HH:MM:SS.fff
                    # check if the confidence score of the current prediction index is above the threshold.
                    if result[np.argmax(result)] > self.threshold:

                        # checks if there are any elements in the sentence list.
                        # If it's not empty, it means there are already recognized actions in the sentence.
                        if len(sentence) > 0:
                            # compares the current predicted action
                            if self.ACTIONS[np.argmax(result)] != sentence[-1]:
                                
                                data = {
                                    "text": self.ACTIONS[np.argmax(result)],
                                    "timestamp": current_time,
                                }
                                sentence.append(data)
                        else:
                            # no recognized actions yet
                            data = {
                                    "text": self.ACTIONS[np.argmax(result)],
                                    "timestamp": current_time,
                                }
                            sentence.append(data)

            # cv2.imshow("MediaPipe Detection", cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        # cv2.destroyAllWindows()
        return sentence
