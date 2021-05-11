import numpy as np
import cv2 as cv
import random
import string
import os
import re


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def save_frames(input_name, output_name):
    print(input_name, output_name)
    cap = cv.VideoCapture(input_name)
    while True:
        ret, frame = cap.read()
        i = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        if ret:
            if (i > 1000 and i < 6000 and i % 60 == 0):
                output = output_name + "/" + get_random_string(8) + ".jpg".format(i)
                cv.imwrite(output, frame)
            if cv.waitKey(1) & 0xFF == ord('q') or i > 6000:
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    inputFolder = r"F:\Dokumente\Uni_Msc\Thesis\videos\Allogymnopleuri_Rolling from dung pat_201611\resized"
    outputFolder = r"F:\Dokumente\Uni_Msc\Thesis\trajectory_extraction\yolo_dataset"
    folderItems = os.listdir(inputFolder)
    folderItems.sort(key=lambda f: int(re.sub('\D', '', f)))
    videos = [fi for fi in folderItems if fi.endswith(".mp4")]
    i = 0

    print(len(videos))

    while i <= len(videos)-1:
        name = inputFolder + "/" + videos[i]
        save_frames(name, outputFolder)

        i += 1
