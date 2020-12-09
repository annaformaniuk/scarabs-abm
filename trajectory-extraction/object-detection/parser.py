import json
import csv
import os
import argparse
import cv2

# python parser.py --json F:\Downloads\database\Allogymnopleuri_#01\Allogymnopleuri_#01_db.grndr --photos_folder F:\Downloads\database\Allogymnopleuri_#01\Allogymnopleuri_#01_imgs --labels Beetle,Ball

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-j", "--json", required=True,
                help="Path to the json with annotations")
ap.add_argument("-f", "--photos_folder", required=True,
                help="Path to the images folder")
ap.add_argument("-l", "--labels", required=True, help="Label of annotations")
args = vars(ap.parse_args())
args["labels"] = args["labels"].split(",")


def get_img_shape(path):
    img = cv2.imread(path)
    try:
        return img.shape
    except AttributeError:
        print('error! ', path)
        return (None)

def write_annotation(num, path, name, position, size):
    print(num, path, name, position, size)
    name_no_ext = os.path.splitext(name)[0]

    full_size = get_img_shape(path + '/' + name)
    if (full_size):
        top_left = [int(i) for i in position.split(';')]
        h_w = [int(j) for j in size.split(';')]
        bottom_right = [top_left[0] + h_w[1], top_left[1] + h_w[0]]
        print(top_left)
        print(bottom_right)
        x = (bottom_right[0] + top_left[0])/2.0
        y = (bottom_right[1] + top_left[1])/2.0

        dw = 1./full_size[1]
        dh = 1./full_size[0]

        x = x*dw
        w = h_w[1]*dw
        y = y*dh
        h = h_w[0]*dh

        out_path = path + '/annotations/' + name_no_ext + '.txt'
        if (os.path.isfile(out_path)):
            with open(out_path, 'a') as fd:
                line = str(num) + ' ' + str(x) + ' ' + \
                    str(y) + ' ' + str(h) + ' ' + str(w)
                fd.write(line)
        else:
            with open(out_path, 'w') as fd:
                line = str(num) + ' ' + str(x) + ' ' + \
                    str(y) + ' ' + str(h) + ' ' + str(w)
                fd.write(line)
                fd.write("\n")

f = open(args['json'])
data = json.load(f)
f.close()

print(args)

for num, labelName in enumerate(args["labels"]):
    # check if label we need is there
    for ref in data['Labels']:
        if (ref['Label']['Name'] == labelName):
                # store references for labels and for image names
            label_references = ref['Label']['ImageBuildPool'][0]['Item']['ImageBuilds']
            image_references = data['ImageReferences']

            # for each label reference, find coordinates of the top left corner,
            # the size of the bounding box
            # and the index of the reference image
            for x in label_references:
                index = None
                position = None
                size = None
                name = None
                # starting with reference for which object is in the image
                if (len(x['ImageBuild']['Layers']) > 0):
                    if (len(x['ImageBuild']['Layers'][0]['Layer']['DraftItems']) > 0):
                        index = x['ImageBuild']['ImageReference']
                        all_properties = x['ImageBuild']['Layers'][0]['Layer']['DraftItems'][0]['DraftItem']['Properties']

                        for prop in all_properties:
                            if (prop['Property']['Name'] == 'Position'):
                                position = prop['Property']['Value']
                            if (prop['Property']['Name'] == 'Size'):
                                size = prop['Property']['Value']
                        if (index and position and size):
                            print(index, position, size)
                            for ref in image_references:
                                if (ref['ImageReference']['Index'] == index):
                                    path = ref['ImageReference']['File']
                                    base_name = os.path.basename(path)
                                    if (base_name):
                                        write_annotation(
                                            num, args['photos_folder'], base_name, position, size)
