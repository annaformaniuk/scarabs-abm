import json
import csv
import os
import argparse
import cv2
import numpy as np
#F:\Git\MSc\CV_DL_Stuff\yolo_custom\next_attempt\darknet\data\obj
# python draw_yolo_coords.py --photos_folder F:\Dokumente\Uni_Msc\Thesis\frames_database\Lamarcki_#09\Lamarcki_#09_imgs --annotations_folder F:\Dokumente\Uni_Msc\Thesis\frames_database\Lamarcki_#09\Lamarcki_#09_txt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-photos_folder", "--photos_folder", required=True,
                help="Path to the images folder")
ap.add_argument("-annotations_folder", "--annotations_folder", required=True,
                help="Path to the folder with annotations")
args = vars(ap.parse_args())


def get_random_color(classid):
    # color = list(np.random.choice(range(256), size=3))
    if (classid == 0):
        color = [200,0,0]
    else:
        color = [0,200,0]
    return color

def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    print("bounding box", x1, y1, x2, y2)
    return x1, y1, x2, y2


folderItems = os.listdir(args["annotations_folder"])
txts = [fi for fi in folderItems if fi.endswith(".txt")]

i = 0

while i < len(txts):
    # read label textfile
    refTxtPath = args["annotations_folder"] + "/" + txts[i]
    name_no_ext = os.path.splitext(txts[i])[0]
    
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

            for line in lines:
                floats = [float(x) for x in line]
                x1, y1, x2, y2 = from_yolo_to_cor(floats[1:], copyImage.shape)

                random_color = get_random_color(floats[0])
                cv2.rectangle(copyImage, (x1, y1), (x2, y2), (int(random_color[0]), int(random_color[1]), int(random_color[2])), 3)


            # percent by which the image is resized
            scale_percent = 50

            # calculate the 50 percent of original dimensions
            width = int(copyImage.shape[1] * scale_percent / 100)
            height = int(copyImage.shape[0] * scale_percent / 100)
            # dsize
            dsize = (width, height)

            # resize image
            output = cv2.resize(copyImage, dsize)

            cv2.imshow(name_no_ext, output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    i += 1