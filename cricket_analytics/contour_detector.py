import cv2
import numpy as np

class ContourDetector:
    def __init__(self, min_area=5, min_circularity=0.4, kernel_size=(5, 5)):
        self.min_area = min_area
        self.min_circularity = min_circularity
        self.kernel = np.ones(kernel_size, np.uint8)

    def apply_morphological_operations(self, mask):
        """Apply erosion and dilation to the mask."""
        mask_eroded = cv2.erode(mask, self.kernel, iterations=1)
        mask_dilated = cv2.dilate(mask_eroded, self.kernel, iterations=1)
        return mask_dilated

    def filter_contours(self, contours):
        """Filter contours based on area and circularity."""
        centroids = []
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                
                if circularity >= self.min_circularity:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx = (x + x + w) // 2
                    cy = (y + y + h) // 2
                    centroids.append((cx, cy))
                    filtered_contours.append(contour)
        
        return filtered_contours, centroids

    def detect(self, mask):
        """Perform contour detection with morphological operations and filtering."""
        morph_mask = self.apply_morphological_operations(mask)
        contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self.filter_contours(contours)
    
    def draw_contours(self, frame, contours, centroids, draw_contours=True, draw_centroids=True):
        """Draw contours and centroids on the frame based on user preference."""
        if draw_contours:
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Green contours
        
        if draw_centroids:
            for cx, cy in centroids:
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # Red centroids
        
        return frame
