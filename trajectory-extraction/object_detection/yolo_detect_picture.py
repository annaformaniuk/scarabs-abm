import cv2
import numpy as np
import time


class Yolo_detector():
    def __init__(self):
        print("Initializing Yolo detector class")

        self.CONFIDENCE = 0.5
        self.SCORE_THRESHOLD = 0.5
        self.IOU_THRESHOLD = 0.5

        # the neural network configuration
        config_path = r"F:\Git\MSc\CV_DL_Stuff\yolo_custom\next_attempt\darknet\cfg\yolov3-custom.cfg"
        # the YOLO net weights file
        weights_path = r"F:\Git\MSc\CV_DL_Stuff\yolo_custom\next_attempt\darknet\backup\yolov3-custom_final.weights"

        # loading all the class labels (objects)
        self.labels = open(r"F:\Git\MSc\CV_DL_Stuff\yolo_custom\next_attempt\darknet\data\obj.names").read(
        ).strip().split("\n")
        # generating colors for each object for later plotting
        #colors = np.random.randint(
            #0, 255, size=(len(labels), 3), dtype="uint8")

        # load the YOLO network
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    def detect_objects(self, image):
        h, w = image.shape[:2]
        # create 4D blob
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (416, 416), swapRB=True, crop=False)

        # sets the blob as the input of the network
        self.net.setInput(blob)

        # get all the layer names
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # feed forward (inference) and get the network output
        # measure how much it took in seconds
        start = time.perf_counter()
        layer_outputs = self.net.forward(ln)
        time_took = time.perf_counter() - start
        print(f"Time took to detect objects: {time_took:.2f}s")

        boxes, confidences, class_ids = [], [], []

        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the object detections
            for detection in output:
                # extract the class id (label) and confidence (as a probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # discard weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.CONFIDENCE:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # perform the non maximum suppression given the scores defined before
        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, self.SCORE_THRESHOLD, self.IOU_THRESHOLD)

        objects = []
    
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                bounding_box = [x, y, x + w, y + h]
                objects.append({
                    "label": self.labels[class_ids[i]],
                    "box": bounding_box,
                    "confidence": confidences[i]
                })

        return objects