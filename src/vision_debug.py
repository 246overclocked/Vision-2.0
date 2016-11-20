#
# The main vision program to be run when the Jetson starts up. This
# version will launch with a window containing a set of trackbars where
# you can adjust the bounds of HSV color filtering.
#

import cv2
import cv
import numpy as np
import VisionUtilities
from PIL import Image
from PIL import ImageTk
import Tkinter as tki
import tkFileDialog
import FileDialog
import os
import colorsys
from Tkinter import Frame

__author__ = "Jacob Nazarenko"
__email__ = "jacobn@bu.edu"
__license__ = "MIT"


class BlobDetector():

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
        cv2.createTrackbar('HU', 'HSV', 180, 180, nothing)
        cv2.createTrackbar('SU', 'HSV', 255, 255, nothing)
        cv2.createTrackbar('VU', 'HSV', 255, 255, nothing)
        self.hl = 0
        self.sl = 0
        self.vl = 0
        self.hu = 0
        self.su = 0
        self.vu = 0

        self.window_thread = VisionUtilities.StoppableThread(target=self.window_runner)
        self.window_thread.start()

        self.stopped = False

        self.root = tki.Tk()
        self.top = Frame(self.root)
        self.bottom = Frame(self.root)
        self.top.pack(side='top')
        self.bottom.pack(side='bottom', fill='both', expand=True)
        self.panel_left = None
        self.panel_right = None
        save_btn = tki.Button(self.root, text="Open Config", command=lambda: self.file_open())
        open_btn = tki.Button(self.root, text="Save Config", command= lambda: self.file_save())
        snapshot_btn = tki.Button(self.root, text="Take Snapshot")  # TODO add 'save snapshot' function
        save_btn.pack(in_=self.bottom, side="left", fill="both", expand="yes", padx=10, pady=10)
        open_btn.pack(in_=self.bottom, side="left", fill="both", expand="yes", padx=10, pady=10)
        snapshot_btn.pack(in_=self.bottom, side="left", fill="both", expand="yes", padx=10, pady=10)
        self.image_rgb_filtered = None
        self.image_rgb_hulls = np.zeros((600, 800, 3), np.uint8)

    def image_callback(self):

        print("Hello, OpenCV!\nLoading feed...")
        # be sure to enter the correct camera ip here!
        cap = cv2.VideoCapture("http://root:underclocked@128.197.50.90/mjpg/video.mjpg")
        if not cap.isOpened():
            print "ERROR RETRIEVING STREAM!\nExiting..."
            return
        else:
            print "Success!\nAnalyzing stream...     (Press ESC to quit!)"

        while not self.stopped:
            retval, frame = cap.read()
            if not retval:
                print "ERROR READING FRAME!"
                continue

            imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.image_rgb_hulls = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
            COLOR_BOUNDS = [np.array([self.hl, self.sl, self.vl]), np.array([self.hu, self.su, self.vu])]
            self.finalMask = cv2.inRange(imageHSV, COLOR_BOUNDS[0], COLOR_BOUNDS[1])

            # TODO call this version 'debug' and create separate version that will use an unchangeable mask:
            # GREEN_BOUNDS = [np.array([40, 110, 80]), np.array([80, 255, 255])]  # Green filter values
            # RED_BOUNDS = [np.array([0, 180, 80]), np.array([16, 255, 255])]  # Red filter values
            # redMask = cv2.inRange(imageHSV, RED_BOUNDS[0], RED_BOUNDS[1])
            # greenMask = cv2.inRange(imageHSV, GREEN_BOUNDS[0], GREEN_BOUNDS[1])
            # self.finalMask = cv2.add(redMask, greenMask)

            filteredHSV = cv2.bitwise_and(imageHSV, imageHSV, mask=self.finalMask)
            self.image = cv2.cvtColor(filteredHSV, cv2.COLOR_HSV2BGR)
            self.image_rgb_filtered = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            contours, h = cv2.findContours(self.finalMask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # print "Found", len(contours), "contours"
            if len(contours) > 0:
                hulls = [cv2.convexHull(cnt) for cnt in contours]
                hulls = sorted(hulls, key=lambda c: cv2.contourArea(c), reverse=True)

                contour_color = colorsys.hsv_to_rgb(abs(self.hu + self.hl)/360.0, 1, 1)
                cv2.drawContours(self.image_rgb_hulls, hulls, 0,
                                 (contour_color[0]*255, contour_color[1]*255, contour_color[2]*255),
                                 thickness=cv.CV_FILLED)  # draws contour(s)

            try:
                hull_image = Image.fromarray(self.image_rgb_hulls)
                hull_image = ImageTk.PhotoImage(hull_image)
                filtered_image = Image.fromarray(self.image_rgb_filtered)
                filtered_image = ImageTk.PhotoImage(filtered_image)

                if self.panel_right is None or self.panel_left is None:
                    self.panel_left = tki.Label(image=filtered_image)
                    self.panel_left.image = filtered_image
                    self.panel_left.pack(side="left", padx=10, pady=10)
                    self.panel_right = tki.Label(image=hull_image)
                    self.panel_right.image = hull_image
                    self.panel_right.pack(side="right", padx=10, pady=10)

                else:
                    self.panel_left.configure(image=filtered_image)
                    self.panel_left.image = filtered_image
                    self.panel_right.configure(image=hull_image)
                    self.panel_right.image = hull_image

            except RuntimeError:
                print("[INFO] caught a RuntimeError")

            self.root.update_idletasks()
            self.root.update()

        self.window_thread.stop()
        cv2.destroyAllWindows()
        return

    def file_save(self):
        f = tkFileDialog.asksaveasfile(mode='w', defaultextension=".txt")
        if f is None:
            return
        text2save = "hl:" + str(self.hl) + \
                    "\nsl:" + str(self.sl) + \
                    "\nvl:" + str(self.vl) + \
                    "\nhu:" + str(self.hu) + \
                    "\nsu:" + str(self.su) + \
                    "\nvu:" + str(self.vu)
        f.write(text2save)
        f.close()

    def file_open(self):
        f = tkFileDialog.askopenfile(mode='r', defaultextension=".txt")
        if f is None:
            return
        params = f.readlines()
        print params
        try:
            cv2.setTrackbarPos('HL', 'HSV', int(params[0].split(':')[1]))
            cv2.setTrackbarPos('SL', 'HSV', int(params[1].split(':')[1]))
            cv2.setTrackbarPos('VL', 'HSV', int(params[2].split(':')[1]))
            cv2.setTrackbarPos('HU', 'HSV', int(params[3].split(':')[1]))
            cv2.setTrackbarPos('SU', 'HSV', int(params[4].split(':')[1]))
            cv2.setTrackbarPos('VU', 'HSV', int(params[5].split(':')[1]))
        except:
            print "Invalid file structure"
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
        cv2.imshow('HSV', cv2.resize(self.image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA))
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
