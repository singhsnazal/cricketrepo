import cv2
import numpy as np
import os
from background_subtractor import BackgroundSubtractor
from contour_detector import ContourDetector
from hsv_filter import HSVFilter
import cv2
import mediapipe as mp
import os
import warnings
import tensorflow as tf
import re

class YellowObjectTrackerr:
    def __init__(self, video_path, min_area=5, min_circularity=0.4, draw_contours=True, static_folder='static', draw_centroids=True):
        self.cap = cv2.VideoCapture(video_path)
        self.bg_subtractor = BackgroundSubtractor()
        self.contour_detector = ContourDetector(min_area=min_area, min_circularity=min_circularity)
        self.hsv_filter = HSVFilter(lower_bound=[20, 100, 100], upper_bound=[30, 255, 255])
        self.draw_contours = draw_contours
        self.draw_centroids = draw_centroids
        self.static = static_folder
        self.flag=0

        os.makedirs(self.static, exist_ok=True)  # Create the static folder if it doesn't exist

        # List to store centroids for drawing trajectory
        self.centroid_history_before_bounce = []  # Trajectory before bounce (red)
        self.centroid_history_after_bounce = []   # Trajectory after bounce (blue)
        self.previous_velocity = None
        self.bounce_point = None
        self.first_bounce_detected = False  # Flag to check if first bounce has been detected
        self.save_frames_after_bounce = False  # Flag to start saving frames after bounce
                                              
        # Get video frame dimensions for VideoWriter
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define the codec and create a VideoWriter object to save the video
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
        # self.out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (self.frame_width, self.frame_height))  # 30.0 is the frame rate

    def process_frame(self, frame, frame_id):
        fg_mask = self.bg_subtractor.apply(frame)
        
        mask_yellow = self.hsv_filter.apply(frame)
        moving_yellow_mask = cv2.bitwise_and(fg_mask, mask_yellow)
        
        contours, centroids = self.contour_detector.detect(moving_yellow_mask)
        
        if self.draw_contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

        if len(centroids) == 1:
            cx, cy = centroids[0]
            
            # Store the centroid for drawing the trajectory
            if not self.first_bounce_detected:
                self.centroid_history_before_bounce.append((cx, cy))  # Before bounce (red)
            else:
                self.centroid_history_after_bounce.append((cx, cy))  # After bounce (blue)

            # Calculate velocity (change in y position over time)
            if len(self.centroid_history_before_bounce) > 1:
                prev_cx, prev_cy = self.centroid_history_before_bounce[-2]
                velocity = cy - prev_cy  # Vertical velocity (change in y)

                # Detect first bounce
                if not self.first_bounce_detected and self.previous_velocity is not None and velocity < 0 and self.previous_velocity > 0:
                    self.bounce_point = (frame_id, cx, cy)
                    self.first_bounce_detected = True  # Mark the first bounce as detected
                    self.save_frames_after_bounce = True  # Start saving frames
                    print(f"First Bounce detected at Frame {frame_id}: {cx}, {cy}")

                self.previous_velocity = velocity

        return frame
    def roi_detect(self):
        tf.get_logger().setLevel('ERROR')
        warnings.filterwarnings('ignore')

        # Initialize MediaPipe pose model
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        # Define the folder paths
        input_folder = r"static"
        output_folder = r"static"
        os.makedirs(output_folder, exist_ok=True)
        pattern = re.compile(r"bounce_to_hit_frame_\d+\.jpg")


        # Process each frame in the input folder
        for filename in os.listdir(input_folder):
            if pattern.match(filename):
             if filename.endswith(".jpg") or filename.endswith(".png"):
                frame_path = os.path.join(input_folder, filename)
                image = cv2.imread(frame_path)

                # Convert the image to RGB as MediaPipe requires RGB format
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect the player pose
                results = pose.process(image_rgb)

                # Check if pose landmarks were detected
                if results.pose_landmarks:
                    # Initialize bounding box coordinates
                    h, w, _ = image.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0

                    # Calculate the bounding box for the detected pose
                    for landmark in results.pose_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        x_min, y_min = min(x_min, x), min(y_min, y)
                        x_max, y_max = max(x_max, x), max(y_max, y)

                    # Add padding to the bounding box
                    padding = 100
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Extract the ROI of the player
                    player_roi = image[y_min:y_max, x_min:x_max]

                    # Save the segmented player image to the output folder
                    cv2.imwrite(os.path.join(output_folder, f"segmented_{filename}"), player_roi)

        # Close the MediaPipe pose instance
        pose.close()

    def draw_trajectory(self, frame):
        # Define the hit index manually (example: frame 10 after bounce)
        hit_frame_index = 5  # Modify this based on where you believe the hit occurs

        # Draw the trajectory before the bounce (red)
        if len(self.centroid_history_before_bounce) > 1:
            points = np.array(self.centroid_history_before_bounce, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=10)  # Red

        # Draw the trajectory after the bounce up to the hit point (blue)
        if len(self.centroid_history_after_bounce) > 1:
            limited_post_bounce_points = np.array(self.centroid_history_after_bounce[:hit_frame_index], np.int32)
            limited_post_bounce_points = limited_post_bounce_points.reshape((-1, 1, 2))
            
            if len(self.centroid_history_before_bounce) > 0:
                last_pre_bounce_point = self.centroid_history_before_bounce[-1]
                first_post_bounce_point = limited_post_bounce_points[0][0]
                cv2.line(frame, tuple(last_pre_bounce_point), tuple(first_post_bounce_point), color=(255, 0, 0), thickness=10)

            cv2.polylines(frame, [limited_post_bounce_points], isClosed=False, color=(255, 0, 0), thickness=10)

    def run(self, output_path):
        frame_id = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(self.cap.get(3)), int(self.cap.get(4))))

        hit_frame_index = 5  # Frame after the bounce where you want to stop saving frames

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame1 = frame.copy()

            frame_id += 1
            cv2.putText(frame, f"Frame ID: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            processed_frame = self.process_frame(frame, frame_id)
            
            self.draw_trajectory(processed_frame)
            
            if self.flag==0:
               frame_n=0
            if self.save_frames_after_bounce and len(self.centroid_history_after_bounce) <= hit_frame_index:
                
                 cv2.imwrite(f"{self.static}/bounce_to_hit_frame_{frame_n}.jpg", frame1)  # Save original frame
                 frame_n=frame_n+1
                 self.flag=1

            out.write(processed_frame)
            


            
            # cv2.imshow('Yellow Moving Object Detection', processed_frame)
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.roi_detect() 


if __name__ == "__main__":
    tracker = YellowObjectTrackerr(video_path="1.MP4")
    tracker.run(output_path="processed_video.MP4")
