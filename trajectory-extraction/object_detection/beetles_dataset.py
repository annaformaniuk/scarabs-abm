# USAGE
# python deep_learning_with_opencv.py --image F:\Dokumente\Uni_Msc\Thesis\trajectory_extraction\yolo_dataset_test\alfjjmsa.jpg --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os


LABELS = r"F:\Git\MSc\CV_DL_Tutorials\6_deep-learning\synset_words.txt"
PROTOTXT = r"F:\Git\MSc\CV_DL_Tutorials\6_deep-learning\bvlc_googlenet.prototxt"
MODEL = r"F:\Git\MSc\CV_DL_Tutorials\6_deep-learning\bvlc_googlenet.caffemodel"


def find_beetle(input_name, output_name):
	# # construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required=True,
	#                 help="path to input image")
	# ap.add_argument("-p", "--prototxt", required=True,
	#                 help="path to Caffe 'deploy' prototxt file")
	# ap.add_argument("-m", "--model", required=True,
	#                 help="path to Caffe pre-trained model")
	# ap.add_argument("-l", "--labels", required=True,
	#                 help="path to ImageNet labels (i.e., syn-sets)")
	# args = vars(ap.parse_args())
	# args["image"] = r"F:\Dokumente\Uni_Msc\Thesis\trajectory_extraction\yolo_dataset_test\alfjjmsa.jpg"

	# load the input image from disk
	image = cv2.imread(input_name)

	orig_image = image.copy()
	orig_image = cv2.bilateralFilter(orig_image, 9, 75, 75)

	gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

	contours, hierarchy = cv2.findContours(
		thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
	boundRect = [None]*len(cnts)

	# load the class labels from disk
	rows = open(LABELS).read().strip().split("\n")
	classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

	for c in cnts:
		# white_bg = 255*np.ones_like(orig_image)
		x, y, w, h = cv2.boundingRect(c)
		x -= 100
		y -= 100
		w += 200
		h += 200
		if (x > 0 and y > 0):
			# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
			roi = image[y:y + h, x:x + w]
			# white_bg[y:y+h, x:x+w] = roi

			# cv2.imshow('hello', roi)
			# cv2.waitKey(0)

			# our CNN requires fixed spatial dimensions for our input image(s)
			# so we need to ensure it is resized to 224x224 pixels while
			# performing mean subtraction (104, 117, 123) to normalize the input;
			# after executing this command our "blob" now has the shape:
			# (1, 3, 224, 224)
			blob = cv2.dnn.blobFromImage(roi, 1, (224, 224), (104, 117, 123))

			# load our serialized model from disk
			print("[INFO] loading model...")
			net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

			# set the blob as input to the network and perform a forward-pass to
			# obtain our output classification
			net.setInput(blob)
			start = time.time()
			preds = net.forward()
			end = time.time()
			print("[INFO] classification took {:.5} seconds".format(end - start))

			# sort the indexes of the probabilities in descending order (higher
			# probabilitiy first) and grab the top-5 predictions
			idxs = np.argsort(preds[0])[::-1][:5]

			# loop over the top-5 predictions and display them
			for (i, idx) in enumerate(idxs):
					# draw the top prediction on the input image
				# if i == 0:
				# 	text = "Label: {}, {:.2f}%".format(classes[idx],
				# 									preds[0][idx] * 100)
				# 	cv2.putText(roi, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
				# 				0.7, (0, 0, 255), 2)

				# display the predicted label + associated probability to the
				# console
				print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
																		classes[idx], preds[0][idx]))

				if (classes[idx] == 'dung beetle'):
					base = os.path.basename(input_name)
					name = os.path.splitext(base)[0]
					output = output_name + "/" + name + ".jpg"
					cv2.imwrite(output, roi)

			# # display the output image
			# cv2.imshow("Image Classified", roi)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()


if __name__ == '__main__':
    inputFolder = r"F:\Dokumente\Uni_Msc\Thesis\trajectory_extraction\yolo_dataset"
    outputFolder = r"F:\Dokumente\Uni_Msc\Thesis\trajectory_extraction\yolo_dataset_cut"
    folderItems = os.listdir(inputFolder)
    images = [fi for fi in folderItems if fi.endswith(".jpg")]
    i = 0

    print(len(images))

    while i <= len(images)-1:
        name = inputFolder + "/" + images[i]
        find_beetle(name, outputFolder)

        i += 1