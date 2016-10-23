# USAGE: python index.py --dataset dataset --index index.csv
from colordescriptor import ColorDescriptor
import argparse
import glob
import cv2
import cPickle as pickle


if __name__ == '__main__':
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required = False, default='../ImageData/train/data/*/',
		help = "Path to the directory that contains the images to be indexed")
	ap.add_argument("-i", "--index", required = False, default='index_color_hist.txt',
		help = "Path to where the computed index will be stored")
	args = vars(ap.parse_args())

	# initialize the color descriptor
	cd = ColorDescriptor((8, 12, 3))



	dict_features = {}
	# use glob to grab the image paths and loop over them
	for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
		# extract the image ID (i.e. the unique filename) from the image
		# path and load the image itself
		image_id_ext = imagePath[imagePath.rfind("/") + 1:]
		image = cv2.imread(imagePath)

		# describe the image
		features = cd.describe(image)

		# write the features to file
		features = [str(f) for f in features]
		dict_features[imageID] = features
		# output.write("%s,%s\n" % (imageID, ",".join(features)))
	
	# open the output index file for writing
	output = open(args["index"], "w")
	pickle.dump(dict_features, output)

# close the index file
output.close()


# # Go to /../ImageData/train/train_text_tags.txt
# train_tags_path = os.path.join(os.path.dirname(__file__), "..", "ImageData", "train", FILE_TRAIN_INDEX)
