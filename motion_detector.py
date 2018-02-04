# KEYNOTE: This code assumes a static background
# Also this is a motion detection problem rather than object detection
# for color background based tracking we can use Camshift
# for object and structural tracking we can use HIG + LINEAR SVM


# motion sensing and object tarcking using background substraction
# two methods for doing so are
# gaussian mixture based foregraound and background segmentation
# and adaptive badkground mixture with shadow detection

# in motion detection we make assumption that background is mainly static
# therefore if we can model the background we can model the motion

# however in real world this approximation can easily fail
# due to shadowing, changes in lighting, reflections and our
# background can look quite different in different frames
import argparse
import cv2
import imutils
import datetime
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help ="path to the video file")
ap.add_argument("-a", "--min-area", type = int, default = 500, help = "minimum size area")
args = vars(ap.parse_args())

if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
    # time.sleep(0.25)
else:
    camera = cv2.VideoCapture(args["video"])

# inititalizing the first frame
firstFrame = None
# loop over the frames of the video
while True:
    (grabbed, frame) = camera.read()
    text = "Unoccupied"

    # if the frame could not be grabbed then we have reached the end
    if not grabbed:
        break
    # resize the frame, convert it into grayscale image and blur it
    frame = imutils.resize(frame, width = 500)
    # frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # it is important to know that each frame will be different
    # and we use 21*21 convolutional filter to perform blurring
    # this eliminates to smoothen out high frequency noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initilize it
    if firstFrame is None:
        firstFrame = gray
        continue
    # compute the difference between the current frame and static frame
    # delta = background frame - current frame
    # delta = foreground of the image or where motion occurs
    frameDelta = cv2.absdiff(firstFrame, gray)
    # we use thresholding to focous on objects or ROI of image
    # thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilating the threshold image to fill in the holes then
    # find the contours on the image
    # THIS involves convoluting the image with kernal b usually
    # centered around anchor point
    thresh = cv2.dilate(thresh, None, iterations = 2)
    # image is slightly magnified as the maximum pixel intensities are present
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # this returns contours themselves and heirarchy of the contours

    # loop over the contours
    for c in cnts:
        # if the contour is too small ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # compute the bounding box, draw it on the frame and update
        # the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = "Occupied"

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status : {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # SHOW THE FRAME AND RECORD IF THE USER PRESSES A key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("FrameDelta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
# Errors: 1. The Ocuupied sign on the
