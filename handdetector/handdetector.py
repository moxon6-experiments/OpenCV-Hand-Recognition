import cv2
import numpy as np

from .util import dist, get_angle
from .drawables import Hand


class NoHandException(Exception):
    def __init__(self):
        Exception.__init__(self, "No Hand Present in Frame")


class HandDetector:
    def __init__(self):
        self.background_remover = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.background_remover.setNMixtures(3)
        self.background_training = 50

    def start_training(self):
        """Re-initialises training mode after program start"""
        self.__init__()

    def get_hand(self, frame):
        fore = self._train(frame)
        contour = self._get_hand_contour(fore)
        fore = self._clean_foreground_image(fore, contour)
        defects = self._extract_convexity_defects(contour)
        palm_center, palm_radius = self._get_palm_circle(contour, fore)
        finger_points = self._get_finger_points(contour, defects)
        finger_points = self._extract_valid_finger_points(finger_points, palm_center, palm_radius)
        finger_points = self._merge_fingertips(finger_points)
        return Hand(palm_center, palm_radius, finger_points, contour)

    def _train(self, frame):
        """
        Trains background remover during training mode:
            Training area should be empty during this time
        Extracts foreground of frame after training

        Args:
            frame (ndarray) : Current frame to train

        Returns:
            ndarray: Returns extracted foreground
        """
        self.draw_out = np.zeros_like(frame)
        if self.background_training:
            self.background_remover.apply(frame, learningRate=1)
            self.background_training -= 1
            raise NoHandException
        else:
            fore = self.background_remover.apply(frame, learningRate=0)
            fore = cv2.GaussianBlur(fore, (3, 3), 3)
            fore = cv2.erode(fore, np.zeros((3, 3)))
            fore = cv2.dilate(fore, np.zeros((3, 3)))
            ret, fore = cv2.threshold(fore, 127, 255, cv2.THRESH_BINARY)
            return fore

    @staticmethod
    def _get_hand_contour(fore):
        contours = cv2.findContours(fore.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        processed_contours = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area >= 5000:
                processed_contours.append([contour_area, contour])
        processed_contours.sort(key=lambda x: x[0])
        if len(processed_contours) > 0:
            return processed_contours[0][1]
        else:
            raise NoHandException

    @staticmethod
    def _clean_foreground_image(fore, contour):
        fore.fill(0)
        cv2.drawContours(fore, [contour], -1, (255, 255, 255), -1)
        return fore

    @staticmethod
    def _extract_convexity_defects(contour):
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 0:
            defects = cv2.convexityDefects(contour, hull)
            defects = defects.reshape(-1, 4)
            return defects
        else:
            raise NoHandException

    @staticmethod
    def _get_palm_circle(contour, fore):
        dist_max = np.zeros((fore.shape[0], fore.shape[1]))
        for y in range(0, fore.shape[0], 4):
            for x in range(0, fore.shape[1], 4):
                if fore[y, x]:
                    dist_max[y, x] = cv2.pointPolygonTest(contour, (x, y), True)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_max)
        return max_loc, max_val

    @staticmethod
    def _get_finger_points(contour, defects):
        finger_points = []
        for start_index, end_index, far_index, fixpt_depth in defects:
            start_point = contour[start_index][0]  # Left of Finger
            far_point = contour[far_index][0]  # Tip of Finger
            end_point = contour[end_index][0]  # Right of Finger
            finger_points.append([start_point, far_point, end_point])
        return np.array(finger_points)

    @staticmethod
    def _extract_valid_finger_points(finger_points, palm_center, palm_radius):
        processed_finger_points = []
        for start_point, far_point, end_point in finger_points[1:]:
            finger_length = dist(far_point, end_point)  # Length of finger
            if finger_length > palm_radius * 0.9 and (dist(start_point, palm_center) > palm_radius*0.9):
                if get_angle(start_point - far_point, end_point - far_point) < 80:
                    processed_finger_points.append([start_point, far_point, end_point])
        return processed_finger_points

    @staticmethod
    def _merge_fingertips(processed_finger_points):
        for i in range(len(processed_finger_points)-1):
            t1 = processed_finger_points[i][2]
            t2 = processed_finger_points[i+1][0]
            t = np.mean([t1, t2], axis=0).astype(int)
            processed_finger_points[i][-1] = t
            processed_finger_points[i+1][0] = t
        return processed_finger_points
