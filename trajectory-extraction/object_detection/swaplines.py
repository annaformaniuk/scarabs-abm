import json
import csv
import os
import argparse
import cv2
import numpy as np

# python swaplines.py --annotations_folder F:\Dokumente\Uni_Msc\Thesis\frames_database\Ambiguus_#09\Ambiguus_#09_txt_test

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-annotations_folder", "--annotations_folder", required=True,
                help="Path to the images folder with annotations")
args = vars(ap.parse_args())

# https://stackoverflow.com/a/41753038
def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]

folderItems = os.listdir(args["annotations_folder"])
txts = [fi for fi in folderItems if fi.endswith(".txt")]

i = 0

while i < len(txts):
    # read label textfile
    refTxtPath = args["annotations_folder"] + "/" + txts[i]
    name_no_ext = os.path.splitext(txts[i])[0]

    print('reading ', refTxtPath)
    
    with open(refTxtPath, 'r+') as f:
        lines = []
        linestrings = [line.rstrip('\n') for line in f]
       
        if (len(linestrings) == 2):
            print("swapping from", linestrings)
            first_formatted = replacer(linestrings[0], "1", 0)
            lines.append(first_formatted)
            second_formatted = replacer(linestrings[1], "0", 0)
            lines.append(second_formatted)

            print("swapping to", lines)
        
            f.seek(0)
            f.write(str(lines[1]))
            f.write("\n")
            f.write(str(lines[0]))
            f.truncate()

    i += 1