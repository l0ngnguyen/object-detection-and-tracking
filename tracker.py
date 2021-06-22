import math
import cv2


class EuclideanDistTracker:
    def __init__(self, distance_threshold):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        # distance threshold that is the same object
        self.distance_threshold = distance_threshold
    
    def set_threshold(self, new_threshold):
        self.distance_threshold = new_threshold

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.distance_threshold:
                    self.center_points[id] = (cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


class CVMultiTracker:
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                     'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    def __init__(self, tracker_type, **kwargs):
        if tracker_type not in self.tracker_types:
            raise Exception('Không tồn tại kiểu tracking này')
        self.tracker_type = tracker_type
        self.kwargs = kwargs
        self.multi_tracker = cv2.legacy.MultiTracker_create()
        self.object_bboxs = []  # [x, y, w, h]
        self.object_labels = []

    def create_tracker(self):
        kwargs = self.kwargs
        if self.tracker_type == self.tracker_types[0]:
            tracker = cv2.legacy.TrackerBoosting_create(**kwargs)
        elif self.tracker_type == self.tracker_types[1]:
            tracker = cv2.legacy.TrackerMIL_create(**kwargs)
        elif self.tracker_type == self.tracker_types[2]:
            tracker = cv2.legacy.TrackerKCF_create(**kwargs)
        elif self.tracker_type == self.tracker_types[3]:
            tracker = cv2.legacy.TrackerTLD_create(**kwargs)
        elif self.tracker_type == self.tracker_types[4]:
            tracker = cv2.legacy.TrackerMedianFlow_create(**kwargs)
        elif self.tracker_type == self.tracker_types[5]:
            tracker = cv2.legacy.TrackerGOTURN_create(**kwargs)
        elif self.tracker_type == self.tracker_types[6]:
            tracker = cv2.TrackerMOSSE_create(**kwargs)
        elif self.tracker_type == self.tracker_types[7]:
            tracker = cv2.legacy.TrackerCSRT_create(**kwargs)
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in self.tracker_types:
                print(t)
        return tracker

    def update(self, frame):
        ret, bboxs = self.multi_tracker.update(frame)
        self.object_bboxs = list(bboxs)
        res = []
        for bbox, label in zip(self.object_bboxs, self.object_labels):
            x, y, w, h = bbox
            res.append([x, y, w, h, label])

        return res

    def update_all(self, detections):
        pass

    def add(self, frame, bboxs, labels=None):
        n_objects = len(self.object_bboxs)
        if labels is None:
            labels = [str(i) for i in range(n_objects,
                                            n_objects + len(bboxs))]
        for bbox, label in zip(bboxs, labels):
            tracker = self.create_tracker()
            self.multi_tracker.add(tracker, frame, bbox)

            self.object_bboxs.append(bbox)
            self.object_labels.append(label)

    def reset(self):
        self.multi_tracker.clear()
        self.multi_tracker = cv2.legacy.MultiTracker_create()
        self.object_bboxs = []
        self.object_labels = []
