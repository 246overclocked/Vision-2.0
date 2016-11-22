#
# The main vision program to be run when the Jetson starts up. This
# version will launch with a window containing a set of trackbars where
# you can adjust the bounds of HSV color filtering.
#

import cv2
import cv
import numpy as np
import VisionUtilities

__author__ = "Jacob Nazarenko"
__email__ = "jacobn@bu.edu"
__license__ = "MIT"


class BlobDetector:

    """The main class representing the vision detection process. Only one instance needs to be created,
    and all of the graphics and calculations should be taken care of by this instance. """

    def __init__(self):

        def nothing(x):
            pass

        self.image = np.zeros((600, 800, 3), np.uint8)  # sets a default blank image/mask as a placeholders
        self.finalMask = self.image

        cv2.namedWindow('HSV')
        cv2.createTrackbar('HL', 'HSV', 0, 180, nothing)
        cv2.createTrackbar('SL', 'HSV', 0, 255, nothing)
        cv2.createTrackbar('VL', 'HSV', 0, 255, nothing)
        cv2.createTrackbar('HU', 'HSV', 0, 180, nothing)
        cv2.createTrackbar('SU', 'HSV', 0, 255, nothing)
        cv2.createTrackbar('VU', 'HSV', 0, 255, nothing)
        self.hl = 0
        self.sl = 0
        self.vl = 0
        self.hu = 0
        self.su = 0
        self.vu = 0

        self.window_thread = VisionUtilities.StoppableThread(target=self.window_runner)
        self.window_thread.start()

        self.stopped = False

    def image_callback(self):

        print("Hello, OpenCV!\nLoading feed...")
        # be sure to enter the correct camera ip here!
        cap = cv2.VideoCapture("http://root:underclocked@192.168.1.199/mjpg/video.mjpg")
        if not cap.isOpened():
            print "ERROR RETRIEVING STREAM!\nExiting..."
            return
        else:
            print "Success!\nAnalyzing stream..."

        while not self.stopped:
            retval, frame = cap.read()
            if not retval:
                print "ERROR READING FRAME!"
                continue

            imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            COLOR_BOUNDS = [np.array([self.hl, self.sl, self.vl]), np.array([self.hu, self.su, self.vu])]
            self.finalMask = cv2.inRange(imageHSV, COLOR_BOUNDS[0], COLOR_BOUNDS[1])

            # TODO call this version 'debug' and create separate version that will use an unchangeable mask:

            # GREEN_BOUNDS = [np.array([40, 110, 80]), np.array([80, 255, 255])]  # Green filter values
            # RED_BOUNDS = [np.array([0, 180, 80]), np.array([16, 255, 255])]  # Red filter values
            # redMask = cv2.inRange(imageHSV, RED_BOUNDS[0], RED_BOUNDS[1])
            # greenMask = cv2.inRange(imageHSV, GREEN_BOUNDS[0], GREEN_BOUNDS[1])
            # self.finalMask = cv2.add(redMask, greenMask)

            # we may need to erode, but I'll leave this commented out for now:
            # self.finalMask = cv2.erode(self.finalMask, (5,5), iterations=5)

            filteredHSV = cv2.bitwise_and(imageHSV, imageHSV, mask=self.finalMask)
            self.image = cv2.cvtColor(filteredHSV, cv2.COLOR_HSV2BGR)
            contours, h = cv2.findContours(self.finalMask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
            # print "Found", len(contours), "contours"
            hull = []
            if len(contours) > 0:
                hull = cv2.convexHull(contours[0])

            # TODO implement automatic color detection (of center) for drawing
            cv2.drawContours(self.image, hull, -1, (255,0,0), thickness=cv.CV_FILLED)  # draws contour(s)

            # Code for working with multiple vision targets:
            # for contour in contours:
            #     area = cv2.contourArea(contour)
            #     x, y, w, h = cv2.boundingRect(contour)
            #     rect_area = w * h
            #     extent = float(area) / rect_area
            #     if extent > 0 and self.find_area(contour) > 2000:
            #         rect = cv2.minAreaRect(contour)
            #         box = cv2.cv.BoxPoints(rect)
            #         box = np.int0(box)
            #         c = self.find_center(contour)
            #         pos = Point(x=c[0], y=c[1], z=0.0)

            # cv2.waitKey(0)

        self.window_thread.stop()
        cv2.destroyAllWindows()
        return

    def find_area(self, contour):
        moments = cv2.moments(contour)

        return moments['m00']

    def find_center(self, contour):
        moments = cv2.moments(contour)
        x_val = int(moments['m10'] / moments['m00'])
        y_val = int(moments['m01'] / moments['m00'])

        return x_val, y_val

    def window_runner(self):
        cv2.imshow('HSV', cv2.resize(self.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            self.stopped = True
            return 1

        self.hl = cv2.getTrackbarPos('HL', 'HSV')
        self.sl = cv2.getTrackbarPos('SL', 'HSV')
        self.vl = cv2.getTrackbarPos('VL', 'HSV')
        self.hu = cv2.getTrackbarPos('HU', 'HSV')
        self.su = cv2.getTrackbarPos('SU', 'HSV')
        self.vu = cv2.getTrackbarPos('VU', 'HSV')

        return 0


if __name__ == "__main__":
    detector = BlobDetector()
    detector.image_callback()
    cv2.destroyAllWindows()