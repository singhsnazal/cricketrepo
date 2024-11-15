import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from background_subtractor import BackgroundSubtractor
from contour_detector import ContourDetector
from hsv_filter import HSVFilter
pitch_coords = np.array([[189, 671], [522, 663], [715, 865], [3, 880]])

# Define the zones as percentages relative to the height of the trapezium
zones = {
    "Over Pitched": (0, 0.10, (200, 100, 200), "2m"),  # Dimmed purple color
    "Full": (0.10, 0.30, (100, 200, 100), "6m"),        # Dimmed green color
    "Good": (0.30, 0.40, (150, 150, 255), "8m"),        # Dimmed red color
    "Short": (0.40, 1.0, (150, 150, 200), "12m")        # Dimmed blue color
}

# Define the transparency level (0 to 1, where 0 is fully transparent and 1 is opaque)
alpha = 0.5  # Lower value for a softer, dimmed effect


import seaborn as sns

class YellowObjectTracker:
    def __init__(self, video_path, min_area=5, min_circularity=0.4, draw_contours=True, draw_centroids=True, player_min_area=500):
        self.cap = cv2.VideoCapture(video_path)
        self.bounce_detected = False
        self.frame_bounced = False  # Flag to ensure we save only once
        self.bounce_index = None  # To store the frame index where the bounce occurred
        self.bg_subtractor = BackgroundSubtractor()
        self.contour_detector = ContourDetector(min_area=min_area, min_circularity=min_circularity)
        self.hsv_filter = HSVFilter(lower_bound=[10, 100, 100], upper_bound=[90, 255, 255])
        self.draw_contours = draw_contours
        self.draw_centroids = draw_centroids
        self.player_min_area = player_min_area
        self.centroid_positions = []
        self.bounce_detected = False
        self.bounce_index = None
        self.predicted_trajectory = None
        self.frame_ids = []  # To store frame IDs for plotting
        self.cx_list = []    # To store CX values for plotting
        self.cy_list = []    # To store CY values for plotting

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_frame(self, frame, frame_count):
        fg_mask = self.bg_subtractor.apply(frame)
        mask_yellow = self.hsv_filter.apply(frame)
        moving_yellow_mask = cv2.bitwise_and(fg_mask, mask_yellow)
        
        contours, centroids = self.contour_detector.detect(moving_yellow_mask)
        
        if len(centroids) != 1:
            return frame
        
        if self.draw_contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if self.draw_centroids:
            cx, cy = centroids[0]
            # cv2.circle(frame, (int(cx), int(cy)), 12, (0, 0, 255), -1)
            self.centroid_positions.append((cx, cy))
            self.frame_ids.append(frame_count)  # Store frame count
            self.cx_list.append(cx)             # Store CX for plotting
            self.cy_list.append(cy)             # Store CY for plotting
        
        return frame

    # def detect_players(self, frame):
    #     fg_mask = self.bg_subtractor.apply(frame)
    #     kernel = np.ones((5, 5), np.uint8)
    #     fg_mask_dilated = cv2.dilate(fg_mask, kernel, iterations=1)
    #     contours, _ = self.contour_detector.detect(fg_mask_dilated)

    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.pose.process(rgb_frame)

    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area > self.player_min_area:
    #             x, y, w, h = cv2.boundingRect(contour)

    #     if results.pose_landmarks:
    #         self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
    #     return frame
    
    def categorize_bounce(self,distance_in_meters):
        """
        Categorize the bounce point based on its distance in meters.
        
        Parameters:
        distance_in_meters (float): The bounce point distance from the batsman in meters.

        Returns:
        str: Category of the bounce point (e.g., Yorker, Full, Good, Short).
        """
        # Define thresholds in meters (adjust these values according to standard cricket measurements)
        if distance_in_meters <= 2.0:
            return "Yorker"
        elif 2.0 < distance_in_meters <= 6:
            return "Full Length"
        elif 6 < distance_in_meters <= 8:
            return "Good Length"
        elif 8 < distance_in_meters <= 20:
            return "Short Length"
        else:
            return "Bouncer or Out of Range"


    
    def estimate_trajectory(self, frame, frame1, frame_number,fr):
        if len(self.centroid_positions) < 2:
            return
         # Iterate over the centroid positions and draw lines between consecutive points
        for i in range(1, len(self.centroid_positions)):
        # Get the previous and current positions
            prev_pos = self.centroid_positions[i - 1]
            current_pos = self.centroid_positions[i]

            # Draw a line connecting the previous position to the current position
            cv2.line(frame, prev_pos, current_pos, (0, 255, 255), 12)


        if self.bounce_detected and not self.frame_bounced:
            self.frame_bounced = True  # Ensure the bounce is only saved once

            # Save the exact frame where the bounce occurs
            bounce_point = self.centroid_positions[self.bounce_index]
            print("bounce_point", bounce_point)
            cv2.circle(frame1, bounce_point, 10, (0, 255, 255), -1)

            # cv2.circle(frame, bounce_point, 12, (0, 255, 255), -1)
            # bounce_text = f"Bounced Frame - Position: {bounce_point}"

            # Add text at the top-left corner with coordinates
            # cv2.putText(frame, bounce_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Save the exact bouncing frame
            # bounced_frame_filename = f"static/bounced_frame.jpg"
            # cv2.imwrite(bounced_frame_filename, frame1)
            # print(f"Exact bouncing frame saved: {bounced_frame_filename}")

            # Apply perspective transform to get the warped frame
            src_points = np.float32([(189, 671), (522, 663), (715, 865), (3, 880)])  # Replace with actual source points
            dst_points = np.float32([(0, 0), (600, 0), (600, 800), (0, 800)])

            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_frame = cv2.warpPerspective(fr, M, (600, 800))

            # Transform the bounce point into the warped frame
            bounce_point_arr = np.array([[[bounce_point[0], bounce_point[1]]]], dtype="float32")
            warped_bounce_point = cv2.perspectiveTransform(bounce_point_arr, M)
            warped_bounce_point = tuple(map(int, warped_bounce_point[0][0]))

            # Draw the warped bounce point on the warped frame
            cv2.circle(warped_frame, warped_bounce_point, 12, (0, 255,255), -1)

            # Save the warped bouncing frame
            # warped_bounced_frame_filename = f"static/warped_bounced_frame.jpg"
            # cv2.imwrite(warped_bounced_frame_filename, warped_frame)
            # print(f"Exact warped bouncing frame saved: {warped_bounced_frame_filename}")

            # Optionally display the warped frame
            # cv2.imshow("Warped Perspective", warped_frame)
            # cv2.waitKey(1)  # Adjust as needed to keep the window open
            # cv2.destroyAllWindows()

            # Calculate the depth of the ball in meters based on y-coordinate in warped frame
            x, y = warped_bounce_point
            depth_of_ball_in_meters = (20 / 800) * y
            # Updated text with two decimal places and modified position, font size, and thickness
            # bounce_text = f"Bounce Impact Depth: {depth_of_ball_in_meters:.2f} meters"

            # Set a larger font size, bold thickness, and shadow effect for better visibility
            font_scale = 1.0  # Increase font size
            font_thickness = 3  # Make it bolder

            # Add a shadow by drawing black text slightly offset behind the main text
            # cv2.putText(frame1, bounce_text, (150, 770), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)

            # Draw the main text in red on top of the shadow
            # cv2.putText(frame1, bounce_text, (150, 770), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

            # print("Depth of ball in meters:", depth_of_ball_in_meters)
            # bounce_text = f"Pitch_Point: { depth_of_ball_in_meters}"
            # # Format depth with two decimal places
            # bounce_text = f"Pitch_Contact_Depth:{depth_of_ball_in_meters:.2f} m"

            # # Use thickness=3 for bold text
            # cv2.putText(frame1, bounce_text, (312,740), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
            
            bounced_frame_filename = f"static/bounced_frame.jpg"
            cv2.imwrite(bounced_frame_filename, frame1)


            # Add text at the top-left corner with coordinates
            # cv2.putText(frame, bounce_text, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            bounce_category = self.categorize_bounce(depth_of_ball_in_meters)
            font_scale = 1.0  # Increase font size
            font_thickness = 3  # Make it bolder
            text = f"Pitched Category: {bounce_category}" 
            # Add a shadow by drawing black text slightly offset behind the main text
            # cv2.putText( warped_frame, text, (105, 62), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)

            # Draw the main text in red on top of the shadow
            # cv2.putText( warped_frame, text, (105, 62), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
            # print(f"The bounce point is categorized as: {bounce_category}")

            # Combine label and value

            # Display the category on the warped frame
            # cv2.putText(warped_frame, text, (105, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            # Optionally, save the warped frame with the category text
            # cv2.imwrite(f"static/optimal_bounced_frame.jpg", warped_frame)

        else:
            # Reset the bounce state if necessary (e.g., if bounce has been processed)
            if not self.bounce_detected:
                self.frame_bounced = False

        bounce_index = self.detect_bounce()
        print("bounce",bounce_index)
        if bounce_index is not None:
            self.bounce_detected = True
            self.bounce_index = bounce_index
        # self.draw_trajectory_before_bounce(frame)

    def detect_bounce(self):
        for i in range(1, len(self.centroid_positions) - 1):
            prev_y = self.centroid_positions[i - 1][1]
            curr_y = self.centroid_positions[i][1]
            next_y = self.centroid_positions[i + 1][1]
            if curr_y > prev_y and next_y < curr_y:
                # print("framenu",i)
                return i 
        return None
  

    # def draw_trajectory_before_bounce(self, frame):
    #     for i in range(1, len(self.centroid_positions)):
    #         start_point = self.centroid_positions[i - 1]
    #         end_point = self.centroid_positions[i]
            # cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # def predict_trajectory_after_bounce(self, frame):
    #     if self.predicted_trajectory is not None:
    #         # cv2.line(frame, self.predicted_trajectory[0], self.predicted_trajectory[1], (255, 0, 0), 2)
    #         return

    #     start_point = self.centroid_positions[self.bounce_index]
    #     end_point = self.centroid_positions[-1]

    #     dx = end_point[0] - start_point[0]
    #     dy = end_point[1] - start_point[1]
        
    #     if dx == 0:
    #         return
        
    #     slope = dy / dx
    #     intercept = start_point[1] - slope * start_point[0]
        
    #     extended_end_x = end_point[0] + 12 * dx
    #     extended_end_y = int(slope * extended_end_x + intercept)
    #     extended_end_point = (int(extended_end_x), extended_end_y)

    #     cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    #     cv2.line(frame, end_point, extended_end_point, (255, 0, 0), 2)
    #     self.predicted_trajectory = (end_point, extended_end_point)
    
   

    def plot_and_save_graphs(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        # plt.style.use('seaborn-darkgrid')
        plt.style.use('ggplot')

        # Enhancing the plots for a professional and visually appealing look
        # plt.style.use('seaborn-darkgrid')  # A modern, cleaner style

        # Ball Lateral Movement Over Time
        plt.figure(figsize=(8, 6))
        plt.plot(self.frame_ids, self.cx_list, marker='o', markersize=8, color='#0066cc', linestyle='-', linewidth=2.5, alpha=0.85,
                markerfacecolor='#3399ff', markeredgewidth=2, markeredgecolor='#004080', label='Lateral Movement (CX)')
        plt.fill_between(self.frame_ids, self.cx_list, color='#0066cc', alpha=0.15)  # Fill to make it look smoother
        plt.xlabel("Time (seconds)", fontsize=14, fontweight='bold')
        plt.ylabel("Lateral Position (CX)", fontsize=14, fontweight='bold')
        plt.title("Ball Lateral Movement Over Time", fontsize=16, fontweight='bold', color='#333333')
        plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
        plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig("static/frame_id_vs_cx.png", dpi=300)  # Higher resolution for better quality
        plt.close()

        # Ball Vertical Movement Over Time
        plt.figure(figsize=(8, 6))
        plt.plot(self.frame_ids, self.cy_list, marker='^', markersize=8, color='#33cc33', linestyle='-', linewidth=2.5, alpha=0.85,
                markerfacecolor='#66ff66', markeredgewidth=2, markeredgecolor='#248f24', label='Vertical Movement (CY)')
        plt.fill_between(self.frame_ids, self.cy_list, color='#33cc33', alpha=0.15)  # Smoother fill
        plt.xlabel("Time (seconds)", fontsize=14, fontweight='bold')
        plt.ylabel("Height Position (CY)", fontsize=14, fontweight='bold')
        plt.title("Ball Vertical Movement Over Time", fontsize=16, fontweight='bold', color='#333333')
        plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
        plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig("static/frame_id_vs_cy.png", dpi=300)
        plt.close()

        # Ball Trajectory Path
        plt.figure(figsize=(8, 6))
        plt.plot(self.cx_list, self.cy_list, marker='s', markersize=8, color='#ff4d4d', linestyle='-', linewidth=2.5, alpha=0.85,
                markerfacecolor='#ff6666', markeredgewidth=2, markeredgecolor='#cc0000', label='Trajectory Path')
        plt.fill_between(self.cx_list, self.cy_list, color='#ff4d4d', alpha=0.1)  # Slight fill for a gradient effect
        plt.xlabel("Lateral Position (CX)", fontsize=14, fontweight='bold')
        plt.ylabel("Height Position (CY)", fontsize=14, fontweight='bold')
        plt.title("Ball Trajectory Path", fontsize=16, fontweight='bold', color='#333333')
        plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
        plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig("static/cx_vs_cy.png", dpi=300)
        plt.close()

    #     sns.set(style="whitegrid")  # Use seaborn style for better aesthetics

    #    # Ball Lateral Movement Over Time
    #     plt.figure(figsize=(6, 6))
    #     plt.plot(self.frame_ids, self.cx_list, marker='o', markersize=6, color='#1f77b4', linestyle='-', linewidth=2, alpha=0.8)
    #     plt.xlabel("Time (seconds)", fontsize=12)
    #     plt.ylabel("Lateral Position (CX)", fontsize=12)
    #     plt.title("Ball Lateral Movement Over Time", fontsize=14, fontweight='bold')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig("static/frame_id_vs_cx.png")
    #     plt.close()

    #     # Ball Vertical Movement Over Time
    #     plt.figure(figsize=(6, 6))
    #     plt.plot(self.frame_ids, self.cy_list, marker='^', markersize=6, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.8)
    #     plt.xlabel("Time (seconds)", fontsize=12)
    #     plt.ylabel("Height Position (CY)", fontsize=12)
    #     plt.title("Ball Vertical Movement Over Time", fontsize=14, fontweight='bold')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig("static/frame_id_vs_cy.png")
    #     plt.close()

    #     # Ball Trajectory Path
    #     plt.figure(figsize=(6, 6))
    #     plt.plot(self.cx_list, self.cy_list, marker='s', markersize=6, color='#d62728', linestyle='-', linewidth=2, alpha=0.8)
    #     plt.xlabel("Lateral Position (CX)", fontsize=12)
    #     plt.ylabel("Height Position (CY)", fontsize=12)
    #     plt.title("Ball Trajectory Path", fontsize=14, fontweight='bold')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.savefig("static/cx_vs_cy.png")
    #     plt.close()



    def run(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(self.cap.get(3)), int(self.cap.get(4))))
        
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            overlay = frame.copy()

        # Calculate the height of the trapezium from top to bottom
            pitch_height = max(pitch_coords[:, 1]) - min(pitch_coords[:, 1])

            # Iterate over the zones to draw each as a trapezoid
            for zone_name, (start_ratio, end_ratio, color, zone_range) in zones.items():
                # Determine the Y-values for the start and end of each zone
                start_y = int(min(pitch_coords[:, 1]) + pitch_height * start_ratio)
                end_y = int(min(pitch_coords[:, 1]) + pitch_height * end_ratio)
                
                # Calculate the corresponding top and bottom line coordinates for each zone
                top_left = [int(pitch_coords[0][0] + (pitch_coords[3][0] - pitch_coords[0][0]) * start_ratio), start_y]
                top_right = [int(pitch_coords[1][0] + (pitch_coords[2][0] - pitch_coords[1][0]) * start_ratio), start_y]
                bottom_left = [int(pitch_coords[0][0] + (pitch_coords[3][0] - pitch_coords[0][0]) * end_ratio), end_y]
                bottom_right = [int(pitch_coords[1][0] + (pitch_coords[2][0] - pitch_coords[1][0]) * end_ratio), end_y]

                # Create the polygon for the zone
                zone_polygon = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                
                # Draw the colored trapezoid with transparency
                cv2.fillPoly(overlay, [zone_polygon], color)
                
                # Optionally, add the zone name (comment out if not needed)
                label_position = (top_left[0] + 10, start_y + 20)
                # Change to a more professional font and color, and increase boldness
                cv2.putText(overlay, zone_name, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Add the meter range text on the right side of the image
                range_position = (frame.shape[1] - 200, start_y + 20)  # Adjust to position text correctly on the right
                cv2.putText(overlay, zone_range, range_position, cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Blend the overlay with the original image using transparency
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Write the processed frame to the output video
            # out.write(frame)

            # Display the result
            # cv2.imshow('Trapezoidal Segmented Pitch', frame)

            # # Exit the video display with 'q' key
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release the video capture and writer objects, and close all windows
        # cap.release()
        # out.release()
        # cv2.destroyAllWindows()

            cv2.imwrite("frameee.jpg",frame)
            fr=cv2.imread("frameee.jpg")

            frame1=frame.copy()
            # cv2.imshow("jdfjfkds",frame1)
            text = f"frame no: {frame_count}" 
            cv2.putText(frame, text, (105, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
         
            processed_frame = self.process_frame(frame, frame_count)
            # Assuming you have a frame number (e.g., `frame_number` variable)
            self.estimate_trajectory(processed_frame, frame1, frame_count,fr)

            # player_detected_frame = self.detect_players(frame)

            # Highlight the bounced frame
            if self.bounce_index is not None and frame_count == self.bounce_index:
                cv2.putText(processed_frame, "Bounce Detected", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(processed_frame, self.centroid_positions[self.bounce_index], 15, (0, 255, 255), -1)
              

            # Write the frame to the output video
            out.write(processed_frame)
            # cv2.imshow('Yellow Moving Object & Player Detection', processed_frame)
            # # cv2.imwrite("frameee.jpg",frame)

            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

            frame_count += 1

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Plot graphs after processing is complete
        self.plot_and_save_graphs()
       

if __name__ == "__main__":
    tracker = YellowObjectTracker(video_path="1.MP4")
    tracker.run(output_path="processed_video.MP4")
