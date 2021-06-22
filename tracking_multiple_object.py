import cv2
from tracker import CVMultiTracker

# theo dõi nhiều đối tượng sử dụng các hàm có sẵn của opencv
# Đối tượng không được detect tự động mà chọn bằng tay

trackerName = 'MEDIANFLOW'
trackers = CVMultiTracker(trackerName)
cap = cv2.VideoCapture(0)

while cap.isOpened():

    _, frame = cap.read()
    if frame is None:
        break

    #frame = imutils.resize(frame, width=600)
    objects = trackers.update(frame)

    # loop over the bounding boxes and draw them on the frame
    for obj in objects:
        x, y, w, h, label = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), obj[4] 

        cv2.putText(frame, str(label), (x, y - 15),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        colors = []
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROIs("Frame", frame, fromCenter=False,
                             showCrosshair=True)
        trackers.add(frame, box)

    # if you want to reset bounding box, select the 'r' key
    elif key == ord("r"):
        trackers.reset()

        box = cv2.selectROIs("Frame", frame, fromCenter=False,
                             showCrosshair=True)
        trackers.add(frame, box)

    elif key == 27:
        break
cap.release()
cv2.destroyAllWindows()
