import cv2


class DrawAble:
    def draw(self, im):
        raise NotImplementedError


class BlankDrawable(DrawAble):
    def draw(self, im):
        pass

class Fingers(DrawAble):
    def __init__(self, fingers):
        self.fingers = fingers

    def draw(self, im):
        for finger in self.fingers:
            cv2.circle(im, tuple(finger[0]), 10, (255, 0, 0))
            cv2.circle(im, tuple(finger[1]), 10, (0, 255, 0))
            cv2.circle(im, tuple(finger[2]), 10, (0, 0, 255))
            cv2.line(im, tuple(finger[0]), tuple(finger[1]), (255, 255, 0), 1)
            cv2.line(im, tuple(finger[1]), tuple(finger[2]), (0, 255, 255), 1)


class Palm(DrawAble):
    def __init__(self, palm_center, palm_radius):
        self.palm_radius = palm_radius
        self.palm_center = palm_center

    def draw(self, im):
        cv2.circle(im, tuple(self.palm_center), 5, (255, 255, 255), 3)
        cv2.circle(im, tuple(self.palm_center), int(self.palm_radius), (255, 255, 255), 2)


class Hand(DrawAble):
    def __init__(self, palm_center, palm_radius, fingers, contour):
        self.palm = Palm(palm_center, palm_radius)
        self.fingers = Fingers(fingers)
        self.contour = Contour(contour)

    def draw(self, im):
        self.contour.draw(im)
        self.palm.draw(im)
        self.fingers.draw(im)


class Contour(DrawAble):
    def __init__(self, contour):
        self.contour = contour

    def draw(self, im):
        cv2.drawContours(im, [self.contour], -1, (255, 0, 0), -1)
        cv2.drawContours(im, [self.contour], -1, (0, 0, 255), 2)


