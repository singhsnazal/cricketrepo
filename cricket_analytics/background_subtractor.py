import cv2

class BackgroundSubtractor:
    def __init__(self, history=1000, varThreshold=50, detectShadows=True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=varThreshold,
            detectShadows=detectShadows
        )
    
    def apply(self, frame):
        return self.bg_subtractor.apply(frame)
