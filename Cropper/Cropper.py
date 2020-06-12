# Improting Image class from PIL module 
from PIL import Image
import os
from os import listdir
from os.path import isfile, join

original_img_path = "/home/arkhadem/DeltaNN/Accuracy/imageNet/images/original"
resized_img_path = "/home/arkhadem/DeltaNN/Accuracy/imageNet/images/resized"

onlyfiles = [f for f in listdir(original_img_path) if isfile(join(original_img_path, f))]

percentage = 0


print("0% completed!")

for img_idx in range(len(onlyfiles)):
	if((100 * img_idx / len(onlyfiles)) > percentage):
		print(str((100 * img_idx / len(onlyfiles))) + "% completed!")
		percentage = ((img_idx / len(onlyfiles)) * 100) + 1;
	im = Image.open(original_img_path + "/" + onlyfiles[img_idx])
	im = im.resize((227, 227))
	# im.show()
	pixVals = list(im.getdata())
	a_file = open(resized_img_path + "/" + os.path.splitext(onlyfiles[img_idx])[0] + ".txt", "w")
	# print(str(img_idx) + " " + str(len(onlyfiles)) + " " + str(img_idx * 100 / len(onlyfiles)) + " " + original_img_path + "/" + onlyfiles[img_idx])
	if(isinstance(pixVals[0], int) == True):
		# print "Continued"
		continue
	else:
		for pixel in pixVals:
			a_file.write(str(pixel[0]) + " " + str(pixel[1]) + " " + str(pixel[2]) + "\n")
	a_file.close()