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
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    print("bounding box", x1, y1, x2, y2)
    return x1, y1, x2, y2


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
            # copy image not to draw over original file
            copyImage = refImage.copy()
            val_objects = yolo.detect_objects(copyImage)
            print(val_objects)

            for line in lines:
                floats = [float(x) for x in line]
                x1, y1, x2, y2 = from_yolo_to_cor(floats[1:], copyImage.shape)

                # random_color = get_random_color()
                cv2.rectangle(copyImage, (x1, y1), (x2, y2), (0, 0, 200), 5)

            if(len(val_objects) > 0):
                for val_object in val_objects:
                    print(val_object)
                    #  [{'label': 'Ball', 'box': [342, 174, 417, 234]}, {'label': 'Beetle', 'box': [356, 224, 391, 270]}]
                    boxcolor = get_class_color(val_object["label"])
                    cv2.rectangle(copyImage, (val_object["box"][0], val_object["box"][1]), (
                        val_object["box"][2], val_object["box"][3]), boxcolor, 3)
                    #text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                    print(val_object["label"])
                    #text = f"{val_object["label"]}: {val_object["confidence"]:.2f}"
                    text = val_object["label"] + ": " + str(val_object["confidence"])
                    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                    text_offset_x = val_object["box"][0]
                    text_offset_y = val_object["box"][1]
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                    overlay = copyImage.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=boxcolor, thickness=cv2.FILLED)
                    # add opacity (transparency to the box)
                    copyImage = cv2.addWeighted(overlay, 0.6, copyImage, 0.4, 0)
                    cv2.putText(copyImage, text, (val_object["box"][0], val_object["box"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

            filename = name_no_ext + "_val.png"
            cv2.imwrite(filename, copyImage)
    i += 1
