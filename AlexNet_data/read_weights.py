from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


net_data = load("bvlc_alexnet.npy", allow_pickle=True).item()

# print net_data

# print type(net_data['conv6'][0])
print shape(net_data['conv6'][0])

# print type(net_data['conv6'][1])

a_file = open("conv6_weight.txt", "w")

# for in_channel in net_data['fc'][0]:
# 	for out_channel in in_channel:
# 	    a_file.write('{}'.format(out_channel))
# 	    a_file.write(" ")

# a_file.close()

for K1 in net_data['conv6'][0]:
	for K2 in K1:
		for in_ch in K2:
			for out_ch in in_ch:
			    a_file.write('{}'.format(out_ch))
			    a_file.write(" ")

a_file.close()