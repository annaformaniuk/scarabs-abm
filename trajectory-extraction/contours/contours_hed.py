import argparse
import cv2 as cv
import os
import numpy as np

# https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/
# python contours_hed.py --edge-detector hed_model --image frame_6450.jpg

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge-detector", type=str, required=True,
                help="path to OpenCV's deep learning edge detector")
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input image")
args = vars(ap.parse_args())


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


cv.dnn_registerLayer('Crop', CropLayer)


# load our serialized edge detector from disk
print("[INFO] loading edge detector...")
protoPath = os.path.sep.join([args["edge_detector"],
                              "deploy.prototxt"])
modelPath = os.path.sep.join([args["edge_detector"],
                              "hed_pretrained_bsds.caffemodel"])

net = cv.dnn.readNet(protoPath, modelPath)

image = cv.imread(args["image"])
(H, W) = image.shape[:2]

inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                           mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False,
                           crop=False)

net.setInput(inp)
out = net.forward()
out = out[0, 0]
out = cv.resize(out, (image.shape[1], image.shape[0]))
out = 255 * out
out = out.astype(np.uint8)
out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
# con = np.concatenate((image, out), axis=1)
cv.imwrite('het_model_result.jpg', out)
# kWinName = 'Holistically-Nested_Edge_Detection'
# cv.imshow(kWinName, out)
# cv.waitKey(0)
