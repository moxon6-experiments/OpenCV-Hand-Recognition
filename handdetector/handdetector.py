import cv2
import numpy as np
from .util import dist

from .drawables import Hand, BlankDrawable


class HandDetector:
    def __init__(self):
        self.background_remover = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.background_remover.setNMixtures(3)
        self.background_training = 50
        self.hand = BlankDrawable()

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
            fore = self.background_remover.apply(frame, learningRate=1)
            self.background_training -= 1
        else:
            fore = self.background_remover.apply(frame, learningRate=0)
            fore = cv2.GaussianBlur(fore, (3, 3), 3)
            fore = cv2.erode(fore, np.zeros((3, 3)))
            fore = cv2.dilate(fore, np.zeros((3, 3)))
            fore = cv2.dilate(fore, np.zeros((3, 3)))
            ret, fore = cv2.threshold(fore, 127, 255, cv2.THRESH_BINARY)
        return fore

    def _start_training(self):
        """
        Re-initialises training mode after program start
        """
        self.background_remover = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.background_remover.setNMixtures(3)
        self.background_training = 50

    def _process_contour(self, fore, contour):
        """
        Extracts a set of convexity defects from a contour

        Args:
            fore (ndarray): Extracted Foreground Image
            contour
        Returns:
            Hand : Processes defects into a Drawable Hand if one exists
            BlankDrawable : Blank Drawable

        """
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull):
            defects = cv2.convexityDefects(contour, hull)
            defects = np.array([x[0] for x in defects])
            if len(defects) >= 3:
                palm_center, palm_radius, fingers = self._process_defects(fore, contour, defects)
                return Hand(palm_center, palm_radius, fingers, contour)
        return BlankDrawable()

    def _process_defects(self, fore, contour, defects):
        palm_center, palm_radius = self._get_palm_circle(contour, fore)
        finger_points = self._get_finger_points(contour, defects)
        processed_finger_points = []
        for start_point, far_point, end_point in finger_points[1:]:
            finger_length = dist(far_point, end_point)  # Length of finger
            if finger_length > 30 and (dist(start_point, palm_center) > palm_radius*0.9):
                if np.mean([start_point[1], end_point[1]]) < palm_center[1] - palm_radius:
                    processed_finger_points.append([start_point, far_point, end_point])
        processed_finger_points = self._merge_fingertips(processed_finger_points)
        return palm_center, palm_radius, processed_finger_points

    @staticmethod
    def _get_contours(fore):
        contours = cv2.findContours(fore.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        processed_contours = []
        for contour in contours:
            if cv2.contourArea(contour) >= 5000:
                processed_contours.append(contour)
        return processed_contours

    @staticmethod
    def _merge_fingertips(processed_finger_points):
        for i in range(len(processed_finger_points)-1):
            t1 = processed_finger_points[i][2]
            t2 = processed_finger_points[i+1][0]
            t = np.mean([t1, t2], axis=0).astype(int)
            processed_finger_points[i][-1] = t
            processed_finger_points[i+1][0] = t
        return processed_finger_points

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
    def _get_palm_circle(contour, fore):

        dist_max = np.zeros((fore.shape[0], fore.shape[1]))

        cv2.imshow("WINDOW1", fore)

        for y in range(0, fore.shape[0], 4):
            for x in range(0, fore.shape[1], 4):
                if fore[y, x]:
                    dist_max[y, x] = cv2.pointPolygonTest(contour, (x, y), True)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_max)
        return max_loc, max_val

    def get_hand(self, frame):
        fore = self._train(frame)
        processed_contours = self._get_contours(fore)

        if len(processed_contours) == 1:
            fore.fill(0)
            cv2.drawContours(fore, [processed_contours[0]], -1, (255, 0, 0), -1)
            hand = self._process_contour(fore, processed_contours[0])
            return hand
        return BlankDrawable()

    def press_key(self, pressed_key):
        if pressed_key == "r":
            self._start_training()
