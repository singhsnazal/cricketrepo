import cv2
import numpy as np

class HSVFilter:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
    
    def apply(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_frame, self.lower_bound, self.upper_bound)
