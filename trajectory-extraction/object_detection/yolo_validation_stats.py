import cv2
import numpy as np
import csv
import os
import argparse
from yolo_detect_picture import Yolo_detector

# python yolo_validation_stats.py --photos_folder F:\Dokumente\Uni_Msc\Thesis\frames_database\Yolo_Evaluation
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-photos_folder", "--photos_folder", required=True,
                help="Path to the images folder with annotations")
args = vars(ap.parse_args())

font_scale = 1
thickness = 1
stats = {
    "beetle_class": {
        "true_positives": [],
        "false_negatives": [],
        "false_positives": []
    },
    "ball_class": {
        "true_positives": [],
        "false_negatives": [],
        "false_positives": []
    }
}


def get_random_color():
    color = list(np.random.choice(range(256), size=3))
    return color


def get_class_color(classname):
    print(classname)
    if (classname == "Ball"):
        return (200, 0, 0)
    elif (classname == "Beetle"):
        return (0, 200, 0)
    else:
        return (100, 100, 100)


def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x1, y1 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    x2, y2 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    print("bounding box", x1, y1, x2, y2)
    return x1, y1, x2, y2


def bb_intersection_over_union(boxA, boxB):
    print(boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    print(iou)
    return iou


folderItems = os.listdir(args["photos_folder"])
yolo = Yolo_detector()

images = [fi for fi in folderItems if fi.endswith(".png")]
annotations = [fi for fi in folderItems if fi.endswith(".txt")]
i = 0

print(len(annotations), len(images))

while i < len(annotations):
    # read label textfile
    refTxtPath = args["photos_folder"] + "/" + annotations[i]
    name_no_ext = os.path.splitext(annotations[i])[0]

    refImagePath = args["photos_folder"] + "/" + name_no_ext + ".png"
    # if the image exists
    if (os.path.isfile(refImagePath)):
        print("read the image", refImagePath)
        refImage = cv2.imread(refImagePath)
        lines = []
        with open(refTxtPath) as f:
            linestrings = [line.rstrip('\n') for line in f]
            for linestring in linestrings:
                lines.append(linestring.split())

        if (len(lines) > 0):
            ground_truth_beetle = []
            ground_truth_ball = []
            # copy image not to draw over original file
            copyImage = refImage.copy()
            val_objects = yolo.detect_objects(copyImage)
            print(val_objects)

            for j in range(len(lines)):
                floats = [float(x) for x in lines[j]]
                x1, y1, x2, y2 = from_yolo_to_cor(floats[1:], copyImage.shape)
                if (j == 0):
                    ground_truth_beetle = [x1, y1, x2, y2]
                elif (j == 1):
                    ground_truth_ball = [x1, y1, x2, y2]

                # random_color = get_random_color()
                cv2.rectangle(copyImage, (x1, y1), (x2, y2), (0, 0, 200), 5)

            if(len(val_objects) > 0):
                for val_object in val_objects:
                    print(val_object)
                    # first show them on the picture
                    #  [{'label': 'Ball', 'box': [342, 174, 417, 234]}, {'label': 'Beetle', 'box': [356, 224, 391, 270]}]
                    boxcolor = get_class_color(val_object["label"])
                    cv2.rectangle(copyImage, (val_object["box"][0], val_object["box"][1]), (
                        val_object["box"][2], val_object["box"][3]), boxcolor, 3)

                    text = val_object["label"] + ": " + \
                        str(val_object["confidence"])
                    (text_width, text_height) = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                    text_offset_x = val_object["box"][0]
                    text_offset_y = val_object["box"][1]
                    box_coords = ((text_offset_x, text_offset_y),
                                  (text_offset_x + text_width + 2, text_offset_y - text_height))
                    overlay = copyImage.copy()
                    cv2.rectangle(
                        overlay, box_coords[0], box_coords[1], color=boxcolor, thickness=cv2.FILLED)
                    # add opacity (transparency to the box)
                    copyImage = cv2.addWeighted(
                        overlay, 0.6, copyImage, 0.4, 0)
                    cv2.putText(copyImage, text, (val_object["box"][0], val_object["box"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

                    # second, update the validation statistics:
                    # If the IoU is > 0.5, it is considered a True Positive, else it is considered a false positive
                    if (val_object["label"] == "Beetle"):
                        if (len(ground_truth_beetle) > 0):
                            iou = bb_intersection_over_union(
                                ground_truth_beetle, val_object["box"])
                            if (iou > 0.5):
                                stats["beetle_class"]["true_positives"].append(
                                    {"image_name": name_no_ext, "iou": iou, "confidence": val_object["confidence"]})
                            else:
                                stats["beetle_class"]["false_positives"].append(
                                    {"image_name": name_no_ext, "iou": iou, "confidence": val_object["confidence"]})
                        else:
                            stats["beetle_class"]["false_positives"].append(
                                {"image_name": name_no_ext, "iou": iou, "confidence": val_object["confidence"]})
                    elif (val_object["label"] == "Ball"):
                        if (len(ground_truth_ball) > 0):
                            iou = bb_intersection_over_union(
                                ground_truth_ball, val_object["box"])
                            if (iou > 0.5):
                                stats["ball_class"]["true_positives"].append(
                                    {"image_name": name_no_ext, "iou": iou, "confidence": val_object["confidence"]})
                            else:
                                stats["ball_class"]["false_positives"].append(
                                    {"image_name": name_no_ext, "iou": iou, "confidence": val_object["confidence"]})
                        else:
                            stats["ball_class"]["false_positives"].append(
                                {"image_name": name_no_ext, "iou": iou, "confidence": val_object["confidence"]})

            if(len(val_objects) < 2):
                if(any(x["label"] == "Beetle" for x in val_objects) == False and len(ground_truth_beetle) > 0):
                    stats["beetle_class"]["false_negatives"].append(
                        {"image_name": name_no_ext})
                if(any(x["label"] == "Ball" for x in val_objects) == False and len(ground_truth_ball) > 0):
                    stats["ball_class"]["false_negatives"].append(
                        {"image_name": name_no_ext})

            filename = "classified_images/" + name_no_ext + "_val.png"
            cv2.imwrite(filename, copyImage)
    i += 1


print("Beetle", len(stats["beetle_class"]["true_positives"]), len(
    stats["beetle_class"]["false_positives"]), len(stats["beetle_class"]["false_negatives"]))
print(stats["beetle_class"]["false_positives"])
print(stats["beetle_class"]["false_negatives"])
print("Ball", len(stats["ball_class"]["true_positives"]), len(
    stats["ball_class"]["false_positives"]), len(stats["ball_class"]["false_negatives"]))
print(stats["ball_class"]["false_positives"])
print(stats["ball_class"]["false_negatives"])

print("******************")
beetles_precision = len(stats["beetle_class"]["true_positives"])/(len(stats["beetle_class"]["true_positives"]) + len(stats["beetle_class"]["false_positives"]))
beetles_recall = len(stats["beetle_class"]["true_positives"])/(len(stats["beetle_class"]["true_positives"]) + len(stats["beetle_class"]["false_negatives"]))

balls_precision = len(stats["ball_class"]["true_positives"])/(len(stats["ball_class"]["true_positives"]) + len(stats["ball_class"]["false_positives"]))
balls_recall = len(stats["ball_class"]["true_positives"])/(len(stats["ball_class"]["true_positives"]) + len(stats["ball_class"]["false_negatives"]))

print("beetles precision and recall", beetles_precision, beetles_recall)
print("balls precision and recall", balls_precision, balls_recall)
