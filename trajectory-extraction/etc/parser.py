import json
import csv
import os
import argparse
import cv2

# python parser.py --json F:\Dokumente\Uni_Msc\Thesis\frames_database\Garetta_#03\Garetta_#03.grndr --photos_folder F:\Dokumente\Uni_Msc\Thesis\frames_database\Garetta_#03\Garetta_#03_imgs --output_folder F:\Dokumente\Uni_Msc\Thesis\frames_database\Garetta_#03\Garetta_#03_txt --labels "Default label"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-j", "--json", required=True,
                help="Path to the json with annotations")
ap.add_argument("-f", "--photos_folder", required=True,
                help="Path to the images folder")
ap.add_argument("-o", "--output_folder", required=True,
                help="Path to where to save the annotations")
ap.add_argument("-l", "--labels", required=True, help="Label of annotations")
args = vars(ap.parse_args())
args["labels"] = args["labels"].split(",")

def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    print("bounding box", x1, y1, x2, y2)
    return x1, y1, x2, y2

def get_img_shape(path):
    img = cv2.imread(path)
    try:
        return img, img.shape
    except AttributeError:
        print('error! ', path)
        return (None)

def write_annotation(num, path, output_path, name, position, size):
    print(num, path, name, position, size)
    name_no_ext = os.path.splitext(name)[0]

    if (os.path.isfile(path + '/' + name)):

        img, full_size = get_img_shape(path + '/' + name)

        if (full_size):
            top_left = [int(i) for i in position.split(';')]
            h_w = [int(j) for j in size.split(';')]
            bottom_right = [top_left[0] + h_w[0], top_left[1] + h_w[1]]

            # print(top_left)
            # print(bottom_right)

            # imcopy = img.copy()
            # cv2.rectangle(imcopy, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0,225,0), 3)
            # cv2.imshow("image", imcopy)
            # cv2.waitKey(0)

            x = (bottom_right[0] + top_left[0])/2.0
            y = (bottom_right[1] + top_left[1])/2.0

            dw = 1./full_size[1]
            dh = 1./full_size[0]

            x = x*dw
            h = h_w[0]*dw
            y = y*dh
            w = h_w[1]*dh

            # x11, y11, x21, y21 = from_yolo_to_cor([x, y, h, w], imcopy.shape)
            # cv2.rectangle(imcopy, (x11, y11), (x21, y21), (0,0,225), 3)
            # cv2.imshow("image", imcopy)
            # cv2.waitKey(0)

            out_path = output_path + '/' + name_no_ext + '.txt'
            # print(num, x, y, h, w, name_no_ext, out_path)
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
                #  and args["labels"][0] == "Default layer"
                if (len(x['ImageBuild']['Layers']) > 0):
                    if(len(args["labels"]) == 1):
                        
                        # only if there's both and the beetle and the ball, cause otherwise there's
                        # no way to tell which one it is
                        # print(x['ImageBuild']['Layers'])
                        if (len(x['ImageBuild']['Layers'][0]['Layer']['DraftItems']) == 2):
                            for layer_index, layer_item in enumerate(x['ImageBuild']['Layers'][0]['Layer']['DraftItems']):
                                index = x['ImageBuild']['ImageReference']
                                all_properties = layer_item['DraftItem']['Properties']

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
                                                    layer_index, args['photos_folder'], args['output_folder'], base_name, position, size)


                    else:
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
                                                num, args['photos_folder'], args['output_folder'], base_name, position, size)
