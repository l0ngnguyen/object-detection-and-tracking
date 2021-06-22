import cv2
import numpy as np


CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
           "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
           "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
           "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
           "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# Define vehicle class
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]

# get it at https://pjreddie.com/darknet/yolo/
YOLOV3_CFG = 'trained_model/yolov3.cfg'
YOLOV3_WEIGHT = 'trained_model/yolov3.weights'

YOLOV3_WIDTH = 416
YOLOV3_HEIGHT = 416


def detect_objects_by_yolo3(net, image, min_confidence, yolo_w, yolo_h, frame_w, frame_h, classes=None):
    img = cv2.resize(image, (yolo_w, yolo_h))
    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (yolo_w, yolo_h), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    layer_output = net.forward(output_layers)

    boxes = []
    class_ids = []
    confidences = []

    for out in layer_output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence and class_id in VEHICLE_CLASSES:
                print("Object name: " + classes[class_id] +
                      " - Confidence: {:0.2f}".format(confidence * 100))
                center_x = int(detection[0] * frame_w)
                center_y = int(detection[1] * frame_h)
                w = int(detection[2] * frame_w)
                h = int(detection[3] * frame_h)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([int(x), int(y), int(w), int(h)])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    if len(indexes) > 0:
        boxes = [boxes[i] for i in indexes.flatten()]
    return boxes, class_ids, confidences


net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHT)

if __name__ == "__main__":
    cap = cv2.VideoCapture('input/highway.mp4')
    _, frame = cap.read()
    width, height, _ = frame.shape
    net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHT)
    boxes, class_ids, confidences = detect_objects_by_yolo3(net, frame, 0.5, YOLOV3_WIDTH,
                                                            YOLOV3_HEIGHT, width, height, classes=CLASSES)
    print(boxes)
