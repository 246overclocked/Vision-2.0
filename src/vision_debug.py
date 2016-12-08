#
# The main vision program to be run when the Jetson starts up. This
# version will launch with a window containing a set of trackbars where
# you can adjust the bounds of HSV color filtering.
#

import Tkinter as tki
import colorsys
import cv
import numpy as np
import tkFileDialog
import tkMessageBox
from PIL import Image
from PIL import ImageTk
from Tkinter import Frame
import cv2

__author__ = "Jacob Nazarenko"
__email__ = "jacobn@bu.edu"
__license__ = "MIT"


class BlobDetector():

    """The main class representing the vision detection process. Only one instance needs to be created,
    and all of the graphics and calculations should be taken care of by this instance. """

    def __init__(self):

        self.image = np.zeros((600, 800, 3), np.float32)  # sets a default blank image/mask as a placeholders
        self.finalMask = self.image

        self.hl = 0
        self.sl = 0
        self.vl = 0
        self.hu = 180
        self.su = 255
        self.vu = 255
        self.min_area = 0
        self.corner_threshold = 0.01

        self.stopped = False

        self.root = tki.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.top = Frame(self.root)
        self.bottom = Frame(self.root)
        self.top_right = Frame(self.top)
        self.top.pack(side='top')
        self.bottom.pack(side='bottom', fill='both', expand=True)
        self.top_right.pack(side='right')
        self.panel_left = None
        self.panel_right = None

        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))

        # TODO still need to create labels (preferably on the left side) for these sliders!!
        self.hl_slider = tki.Scale(self.root, from_=0, to=180, length=1200, tickinterval=10, orient="horizontal", command=self.slider_callback)
        self.hl_slider.pack(in_=self.top, side="top", fill=None, expand="yes", padx=5, pady=2)
        self.sl_slider = tki.Scale(self.root, from_=0, to=255, length=1200, tickinterval=10, orient="horizontal", command=self.slider_callback)
        self.sl_slider.pack(in_=self.top, side="top", fill=None, expand="yes", padx=5, pady=2)
        self.vl_slider = tki.Scale(self.root, from_=0, to=255, length=1200, tickinterval=10, orient="horizontal", command=self.slider_callback)
        self.vl_slider.pack(in_=self.top, side="top", fill=None, expand="yes", padx=5, pady=2)
        self.hu_slider = tki.Scale(self.root, from_=0, to=180, length=1200, tickinterval=10, orient="horizontal", command=self.slider_callback)
        self.hu_slider.pack(in_=self.top, side="top", fill=None, expand="yes", padx=5, pady=2)
        self.su_slider = tki.Scale(self.root, from_=0, to=255, length=1200, tickinterval=10, orient="horizontal", command=self.slider_callback)
        self.su_slider.pack(in_=self.top, side="top", fill=None, expand="yes", padx=5, pady=2)
        self.vu_slider = tki.Scale(self.root, from_=0, to=255, length=1200, tickinterval=10, orient="horizontal", command=self.slider_callback)
        self.vu_slider.pack(in_=self.top, side="top", fill=None, expand="yes", padx=5, pady=2)
        self.area_slider = tki.Scale(self.root, from_=0, to=5000, label="Min Area", length=500, tickinterval=1000, orient="vertical", command=self.slider_callback)
        self.area_slider.pack(side="right", fill=None, expand="no", padx=5, pady=2)
        self.threshold_slider = tki.Scale(self.root, from_=1, to=100, label="Corner Threshold", length=300, tickinterval=10, orient="vertical", command=self.slider_callback)
        self.threshold_slider.pack(in_=self.top_right, side="right", fill=None, expand="no", padx=40, pady=2)
        self.hu_slider.set(180)
        self.su_slider.set(255)
        self.vu_slider.set(255)

        save_btn = tki.Button(self.root, text="Open Config", command=lambda: self.file_open())
        open_btn = tki.Button(self.root, text="Save Config", command=lambda: self.file_save())
        snapshot_btn = tki.Button(self.root, text="Take Snapshot")  # TODO add 'save snapshot' function
        save_btn.pack(in_=self.bottom, side="left", fill="both", expand="yes", padx=10, pady=3)
        open_btn.pack(in_=self.bottom, side="left", fill="both", expand="yes", padx=10, pady=3)
        snapshot_btn.pack(in_=self.bottom, side="left", fill="both", expand="yes", padx=10, pady=3)

        self.image_rgb_filtered = None
        self.image_rgb_hulls = np.zeros((600, 800, 3), np.float32)
        self.corners = np.zeros(self.image_rgb_hulls.size, self.image_rgb_hulls.dtype)

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

            filteredHSV = cv2.bitwise_and(imageHSV, imageHSV, mask=self.finalMask)
            self.image = cv2.cvtColor(filteredHSV, cv2.COLOR_HSV2BGR)
            self.image_rgb_filtered = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            contours, h = cv2.findContours(self.finalMask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            # corners_found, chessboard_corners = cv2.findChessboardCorners(self.image_rgb_filtered, (8, 7), flags=4)
            # cv2.drawChessboardCorners(self.image_rgb_filtered, (8, 7), chessboard_corners, corners_found)
            #
            # if corners_found:
            #     coordinate_corners = []
            #     for y in range(7):
            #         for x in range(8):
            #             coordinate_corners.append([x, y, 0])
            #
            #     calibration = cv2.calibrateCamera([np.array(coordinate_corners, dtype=np.float32)], [np.array(chessboard_corners, dtype=np.float32)], self.image_rgb_hulls.shape[:2])
            #     print calibration

            if len(contours) > 0:
                hulls = [cv2.convexHull(cnt) for cnt in contours]
                hulls = sorted(hulls, key=lambda c: cv2.contourArea(c), reverse=True)
                contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

                solidity = 0
                try:
                    area = cv2.contourArea(contours[0])
                    hull_area = cv2.contourArea(hulls[0])
                    solidity = float(area) / hull_area
                except ZeroDivisionError:
                    pass

                if cv2.contourArea(hulls[0]) >= self.min_area and 0.4 > solidity > 0.25:
                    contour_color = colorsys.hsv_to_rgb(abs(self.hu + self.hl)/360.0, 1, 1)
                    cv2.drawContours(self.image_rgb_hulls, hulls, 0,
                                     (contour_color[0]*255, contour_color[1]*255, contour_color[2]*255),
                                     thickness=cv.CV_FILLED)  # draws contour(s)

                    # Less precise method for corners:
                    epsilon = 0.01 * cv2.arcLength(hulls[0], True)
                    cv2.drawContours(self.image_rgb_hulls, cv2.approxPolyDP(hulls[0], epsilon, True),
                                      -1, (0, 255, 0), thickness=5)
                    self.corners = cv2.approxPolyDP(hulls[0], epsilon, True)
                    print self.corners

                    # Better method:
                    # self.corners = cv2.goodFeaturesToTrack(cv2.cvtColor(self.image_rgb_hulls, cv2.COLOR_RGB2GRAY),
                    #                                        4, self.corner_threshold, 10)
                    corners = []
                    for corner in self.corners:
                        x, y = corner.ravel()
                        cv2.circle(self.image_rgb_hulls, (x, y), self.image_rgb_hulls.shape[0]/150, (255, 0, 0),
                                   thickness=self.image_rgb_hulls.shape[0]/300)
                        corners.append(corner.ravel())

                    corners = sorted(corners, key=lambda c: c[0])
                    corners[:2] = sorted(corners[:2], key=lambda c: c[1])
                    corners[2:] = sorted(corners[2:], key=lambda c: c[1])
                    corners = np.array(corners, dtype=np.float32)
                    coordinate_corners = np.array([np.array([0, 12, 0]),
                                                   np.array([0, 0, 0]),
                                                   np.array([20, 12, 0]),
                                                   np.array([20, 0, 0])], dtype=np.float32)

                    # calibration = cv2.calibrateCamera([coordinate_corners], [corners], self.image_rgb_hulls.shape[:2])

                    camera_matrix = np.array([[351.96591817, 0., 318.85365638],
                                             [0., 356.36889205, 446.57879869],
                                             [0., 0., 1.]], dtype=np.float32)

                    distortion_coefficients = np.array([[-0.12559246, 0.16263054, -0.03021033, 0.0249903, -0.18684171]], dtype=np.float32)

                    if len(corners) == 4:

                        # rvec, tvec, _ = cv2.solvePnPRansac(coordinate_corners, corners, camera_matrix, distortion_coefficients)
                        _, rvec, tvec = cv2.solvePnP(coordinate_corners, corners, camera_matrix, distortion_coefficients)
                        # print "Rotation Vector:\n" + str(rvec)
                        # print "Translation Vector:\n" + str(tvec) + "\n-------------------------"

                        axis_pts = cv2.projectPoints(np.array([[0, 0, 0], [0, 6, 0], [6, 0, 0], [0, 0, 6]], dtype=np.float32),
                                                     rvec, tvec, camera_matrix, distortion_coefficients)[0]
                        origin = (int(axis_pts[0].ravel()[0]), int(axis_pts[0].ravel()[1]))
                        pt1 = (int(axis_pts[1].ravel()[0]), int(axis_pts[1].ravel()[1]))
                        pt2 = (int(axis_pts[2].ravel()[0]), int(axis_pts[2].ravel()[1]))
                        pt3 = (int(axis_pts[3].ravel()[0]), int(axis_pts[3].ravel()[1]))
                        # print (origin, pt1, pt2, pt3)

                        try:
                            cv2.line(self.image_rgb_hulls, origin, pt1, (0, 255, 0), thickness=3)
                            cv2.line(self.image_rgb_hulls, origin, pt2, (255, 0, 0), thickness=3)
                            cv2.line(self.image_rgb_hulls, origin, pt3, (0, 0, 255), thickness=3)
                        except OverflowError:
                            print "Invalid corner configuration"
                            pass

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
                    self.panel_right.pack(side="left", padx=5, pady=10)

                else:
                    self.panel_left.configure(image=filtered_image)
                    self.panel_left.image = filtered_image
                    self.panel_right.configure(image=hull_image)
                    self.panel_right.image = hull_image

            except RuntimeError:
                print("[INFO] caught a RuntimeError")

            self.root.update_idletasks()
            self.root.update()

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
                    "\nvu:" + str(self.vu) + \
                    "\narea:" + str(self.min_area) + \
                    "\ncorner_threshold:" + str(self.corner_threshold)
        f.write(text2save)
        f.close()

    def file_open(self):
        f = tkFileDialog.askopenfile(mode='r', defaultextension=".txt")
        if f is None:
            return
        params = f.readlines()
        try:
            self.hl_slider.set(int(params[0].split(':')[1]))
            self.sl_slider.set(int(params[1].split(':')[1]))
            self.vl_slider.set(int(params[2].split(':')[1]))
            self.hu_slider.set(int(params[3].split(':')[1]))
            self.su_slider.set(int(params[4].split(':')[1]))
            self.vu_slider.set(int(params[5].split(':')[1]))
            self.area_slider.set(int(params[6].split(":")[1]))
            self.threshold_slider.set(int(params[7].split(":")[1]))
        except:
            print "Invalid file structure"
            return

    def slider_callback(self, new_value):
        try:
            self.hl = self.hl_slider.get()
            self.sl = self.sl_slider.get()
            self.vl = self.vl_slider.get()
            self.hu = self.hu_slider.get()
            self.su = self.su_slider.get()
            self.vu = self.vu_slider.get()
            self.min_area = self.area_slider.get()
            self.corner_threshold = self.threshold_slider.get()/100.0
        except AttributeError:
            pass

    def on_closing(self):
        if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            self.stopped = True

    def find_area(self, contour):
        moments = cv2.moments(contour)

        return moments['m00']

    def find_center(self, contour):
        moments = cv2.moments(contour)
        x_val = int(moments['m10'] / moments['m00'])
        y_val = int(moments['m01'] / moments['m00'])

        return x_val, y_val


if __name__ == "__main__":
    detector = BlobDetector()
    detector.image_callback()
