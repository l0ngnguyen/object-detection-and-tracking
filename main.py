import cv2
from tracker import EuclideanDistTracker
import numpy as np
from model_detection import detect_objects_by_yolo3, CLASSES, net


def empty(x):
    pass


cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars', 640, 100)
cv2.createTrackbar('Distance threshold', "TrackBars", 35, 200, empty)
cv2.createTrackbar('Mask threshold', "TrackBars", 200, 255, empty)
cv2.createTrackbar('Area threshold', "TrackBars", 200, 1000, empty)


def detect_objects_by_contour(mask, area_threshold=100, mask_threshold=200, opening_method=False):
    _, mask = cv2.threshold(mask, mask_threshold, 255, cv2.THRESH_BINARY)

    if opening_method:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for idx, cnt in enumerate(contours):
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > area_threshold:  # and hierarchy[0, idx, 3] == -1:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
    return mask, detections


def select_roi(cap):
    while True:
        success, frame = cap.read()
        if success:
            if IMAGE_SIZE != None:
                frame = cv2.resize(frame, IMAGE_SIZE,
                                   interpolation=cv2.INTER_AREA)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(15)
            if key == ord(' '):
                x, y, w, h = cv2.selectROI('ROI selecting', frame, False)
                cv2.destroyWindow('ROI selecting')
                res = x, y, w, h
                break
            if key == 27:
                res = False
                break
        else:
            print('False to get roi')
            res = False
            break
    del cap
    cv2.destroyWindow('Frame')
    return res


video_path = 'input/large_videos/traffic5.mp4'
IMAGE_SIZE = (1280, 720)

# Create region of interest roi
r = select_roi(cv2.VideoCapture(video_path))
# Create tracker object
tracker = EuclideanDistTracker(35)
# Capture video
cap = cv2.VideoCapture(video_path)
# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=40, detectShadows=True)
#object_detector = cv2.createBackgroundSubtractorKNN()

wait_time = 30
while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break
    if IMAGE_SIZE != None:
        frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    distance_th = cv2.getTrackbarPos('Distance threshold', 'TrackBars')
    mask_th = cv2.getTrackbarPos('Mask threshold', 'TrackBars')
    area_th = cv2.getTrackbarPos('Area threshold', 'TrackBars')

    ########### Select region of interest ###############
    if isinstance(r, tuple):
        x, y, w, h = r
        roi = frame[y:y+h, x:x+w]
    else:
        roi = frame
    height, width, _ = roi.shape

    ########## 1. Object Detection ######################
    mask = object_detector.apply(roi)
    mask, detections = detect_objects_by_contour(
        mask, area_th, mask_th, opening_method=False)
    # detections, class_ids, confidences = detect_objects_by_yolo3(
    #       net, roi, 0.5, 416, 416, width, height, classes=CLASSES)

    ########## 2. Object Tracking #######################
    tracker.set_threshold(distance_th)
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    ########### 3. Show image #############################
    fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)
    cv2.putText(roi, str(int(fps))+' fps', (10, 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("roi", roi)
    cv2.imshow("Mask", mask)
    #cv2.imshow("Frame", frame)

    key = cv2.waitKey(wait_time)
    if key == 27:
        break
    elif key == ord(' '):  # space to stop
        wait_time = 0
    elif key == ord('c'):  # "c" to continue
        wait_time = 30
    elif key == ord('+'):  # "+" to increase video speed
        wait_time -= 5
        wait_time = wait_time if wait_time > 0 else 1
    elif key == ord('-'):  # "-" to decrease video speed
        wait_time += 5
    else:
        continue


cap.release()
cv2.destroyAllWindows()
