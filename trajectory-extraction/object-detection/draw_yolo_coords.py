import json
import csv
import os
import argparse
import cv2

# python draw_yolo_coords.py --photos_folder F:\Downloads\database\Allogymnopleuri_#01\Allogymnopleuri_#01_imgs

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-photos_folder", "--photos_folder", required=True,
                help="Path to the images folder with annotations")
args = vars(ap.parse_args())


def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    return x1, y1, x2, y2


folderItems = os.listdir(args["photos_folder"])
txts = [fi for fi in folderItems if fi.endswith(".txt")]

i = 0

while i < len(txts):
    # read label textfile
    refTxtPath = args["photos_folder"] + "/" + txts[i]
    name_no_ext = os.path.splitext(txts[i])[0]
    
    refImagePath = args["photos_folder"] + "/" + name_no_ext + ".png"
    # if the image exists
    if (os.path.isfile(refImagePath)):
        print("read the image", refImagePath)
        refImage = cv2.imread(refImagePath)
        lines = []
        with open(refTxtPath) as file:
            linestrings = [line.rstrip('\n') for line in file]
            for linestring in linestrings:
                lines.append(linestring.split())
            file.close()

        if (len(lines) > 0):
            print(lines)
        
        # copy image not to draw over original file
        copyImage = refImage.copy()

        for line in lines:
            floats = [float(x) for x in line]
            print(floats)
            x1, y1, x2, y2 = from_yolo_to_cor(floats[1:], copyImage.shape)
            cv2.rectangle(copyImage, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.imshow("image", copyImage)
        cv2.waitKey(0)

    i += 1