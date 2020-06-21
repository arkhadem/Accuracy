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
import csv
import sys

# log_file = open("logs_2.txt", "w")

WEIGHT_ABS_LEN = 1 + 7 # 1 bit sign + 7 bits for weight

WEIGHT_DELTA_LEN = {
	"conv1" : 1,
	"conv2" : 1,
	"conv3" : 1,
	"conv4" : 1,
	"conv5" : 1,
	"fc6" : 1,
	"fc7" : 1,
	"fc8" : 1
}
WEIGHT_DELTA_LEN_LOG = {
	"conv1" : 1,
	"conv2" : 1,
	"conv3" : 1,
	"conv4" : 1,
	"conv5" : 1,
	"fc6" : 1,
	"fc7" : 1,
	"fc8" : 1
}
WEIGHT_NUM_LEN = {
	"conv1" : 5,
	"conv2" : 4,
	"conv3" : 4,
	"conv4" : 4,
	"conv5" : 4,
	"fc6" : 6,
	"fc7" : 6,
	"fc8" : 7
}
WEIGHT_NUM_LEN_LOG = {
	"conv1" : 5,
	"conv2" : 4,
	"conv3" : 4,
	"conv4" : 4,
	"conv5" : 4,
	"fc6" : 6,
	"fc7" : 6,
	"fc8" : 7
}

def sign(value):
	if(value < 0):
		return -1
	return 1

def abs(value):
    return value * sign(value)

def quantization(value, MSB_pos, LSB_pos):
    val_sign = sign(value)
    value = abs(value)
    value = float(int(value*(2**(-1 * LSB_pos))) & int((2**(MSB_pos - LSB_pos))-1)) / float(2**(-1 * LSB_pos))
    return value * val_sign

def to_integer_num(value, MSB_pos, LSB_pos):
	return int(abs(value)*(2**(-1 * LSB_pos))) & int((2**(MSB_pos - LSB_pos))-1)

def to_float_num(value, MSB_pos, LSB_pos):
	return float(value) * float(2 ** LSB_pos)

def merge_sort_single(values): 
  
    if len(values)>1: 
        m = len(values)//2
        left = values[:m] 
        right = values[m:] 
        left = merge_sort_single(left) 
        right = merge_sort_single(right) 
  
        values =[] 
  
        while len(left)>0 and len(right)>0: 
            if left[0]<right[0]: 
                values.append(left[0]) 
                left.pop(0) 
            else: 
                values.append(right[0]) 
                right.pop(0) 
  
        for i in left: 
            values.append(i) 
        for i in right: 
            values.append(i) 
                  
    return values 

def merge_sort(values, i_1d): 
  
    if len(values) != len(i_1d):
        print(str(len(value)) + " " + str(len(i_1d)))

    if len(values)>1:
        m = len(values)//2
        left = values[:m]
        left_i_1d = i_1d[:m]
        right = values[m:]
        right_i_1d = i_1d[m:]
        left, left_i_1d = merge_sort(left, left_i_1d) 
        right, right_i_1d = merge_sort(right, right_i_1d) 
  
        values =[]
        i_1d = [] 
  
        while len(left)>0 and len(right)>0: 
            if left[0]<right[0]: 
                values.append(left[0]) 
                left.pop(0)
                i_1d.append(left_i_1d[0]) 
                left_i_1d.pop(0)
                
            else: 
                values.append(right[0]) 
                right.pop(0)
                i_1d.append(right_i_1d[0]) 
                right_i_1d.pop(0)
  
        for i in left: 
            values.append(i)
        for i in left_i_1d: 
            i_1d.append(i)

        for i in right: 
            values.append(i)
        for i in right_i_1d: 
            i_1d.append(i)
                  
    return values, i_1d

def fc_layer_extractor(writer, net_data, net_data_v1, net_data_v2, layer_name):

	transposed = np.moveaxis(net_data[layer_name][0], [-1], [0])
	transposed_v1 = np.moveaxis(net_data_v1[layer_name][0], [-1], [0])
	transposed_v2 = np.moveaxis(net_data_v2[layer_name][0], [-1], [0])

	o_c_num, i_c_num = shape(transposed)

	# print(layer_name + " data loaded")

	Max_weight = -1;
	Min_weight = 1;
	Max_abs_weight = 0;

	T_input_channel = 4
	T_output_channel = 4
	T_output_size = 8
	T_input_size_const = 40

	T_input_num = T_input_channel * T_output_size * T_output_size;
	T_output_num = T_output_channel * T_output_size * T_output_size;

	num_of_abs_unique_weights = 0;
	num_of_abs_unique_weights_log = 0;
	num_of_delta_unique_weights = 0;
	num_of_delta_unique_weights_log = 0;
	unique_weights = []
	unique_weights_log = []
	unique_weight_repetition = []
	unique_weight_repetition_log = []
	num_of_weight_repetition_cells = 0
	num_of_weight_repetition_cells_log = 0

	num_of_zero_weights = 0

	# for oc_idx, oc in enumerate(net_data[layer_name][1]):
	# 	if(Max_weight < oc):
	# 		Max_weight = oc
	# 	if(Min_weight > oc):
	# 		Min_weight = oc
	# 	if(Max_abs_weight < abs(oc)):
	# 		Max_abs_weight = abs(oc)

	# for oc_idx, oc in enumerate(transposed):
	# 	print ("\r" + str((oc_idx * 100)/ o_c_num) + "% of precision calculation finished "),
	# 	for ic_idx, ic in enumerate(oc):
	# 		if(Max_weight < ic):
	# 			Max_weight = ic
	# 		if(Min_weight > ic):
	# 			Min_weight = ic
	# 		if(Max_abs_weight < abs(ic)):
	# 			Max_abs_weight = abs(ic)

	MSB_pos = 0 #int(log2(Max_abs_weight))
	LSB_pos = -7 #MSB_pos - (WEIGHT_ABS_LEN -1)

	# print("Max: " + str(Max_weight) + " Min: " + str(Min_weight) + " Max abs: " + str(Max_abs_weight) + " MSB position: " + str(MSB_pos) + " LSB position: " + str(LSB_pos))

	# bias manipulation
	for oc in range(len(net_data[layer_name][1])):
		net_data_v1[layer_name][1][oc] = quantization(net_data[layer_name][1][oc], MSB_pos, LSB_pos)
		net_data_v2[layer_name][1][oc] = quantization(net_data[layer_name][1][oc], MSB_pos, LSB_pos)

	weight_file = open(layer_name + "_weight_tiled.txt", "w")
	index_1d_file = open(layer_name + "_index_1d_tiled.txt", "w")
	index_ic_file = open(layer_name + "_index_ic_tiled.txt", "w")

	for T_o_c in range(0, o_c_num, T_output_num):
		# print ("\r" + str((T_o_c * 100)/ o_c_num) + "% of precision calculation finished "),
		# sys.stdout.flush()
		# if(T_o_c != 0):
		# 	break
		for T_i_c in range(0, i_c_num, T_input_num):
			for i_c in range(0, T_input_num):
				# for each input channel in tile, sort all weights corresponding to that channel and write output
				abs_weights = []
				index_1d = []
				if((T_i_c + i_c) >= i_c_num):
					continue
				for o_c in range(0, T_output_num):
					if((T_o_c + o_c) >= o_c_num):
						continue
					transposed_v1[T_o_c + o_c][T_i_c + i_c] = quantization(transposed[T_o_c + o_c][T_i_c + i_c], MSB_pos, LSB_pos)
					transposed_v2[T_o_c + o_c][T_i_c + i_c] = quantization(transposed[T_o_c + o_c][T_i_c + i_c], MSB_pos, LSB_pos)
					if(quantization(transposed[T_o_c + o_c][T_i_c + i_c], MSB_pos, LSB_pos) == 0):
						num_of_zero_weights = num_of_zero_weights + 1
						continue
					abs_weights.append(quantization(abs(transposed[T_o_c + o_c][T_i_c + i_c]), MSB_pos, LSB_pos))
					index_1d.append(o_c)

				abs_weights, index_1d = merge_sort(abs_weights, index_1d)
				
				# sort indices related to unique weights
				unique_weight_indices = []
				first_unique_idx = 0
				for i in range(1, len(abs_weights) + 1):
					if(i == len(abs_weights)) or (abs_weights[i] != abs_weights[first_unique_idx]):
						index_1d[first_unique_idx:i] = merge_sort_single(index_1d[first_unique_idx:i])
						first_unique_idx = i

				last_unique_weight = 0;
				last_unique_weight_log = 0;

				for i in range(0, len(abs_weights)):
					o_c = index_1d[i];

					# V1: using log(delta) for encoding
					if((abs_weights[i] > last_unique_weight_log) and (to_integer_num(abs_weights[i] - last_unique_weight_log, MSB_pos, LSB_pos) > 0)):
						# a new unique weight is found, encode it with either delta or abs approach

						if(log2(to_integer_num(abs_weights[i] - last_unique_weight_log, MSB_pos, LSB_pos)) < (2 ** WEIGHT_DELTA_LEN_LOG[layer_name])):
							# using log delta for encoding weight
							num_of_delta_unique_weights_log = num_of_delta_unique_weights_log + 1
							
							# finding floor(log of delta) and its corresponding weight
							log_delta_floor = int(log2(to_integer_num(abs_weights[i] - last_unique_weight_log, MSB_pos, LSB_pos)))
							floor_weight = to_float_num(to_integer_num(last_unique_weight_log, MSB_pos, LSB_pos) + (2 ** log_delta_floor), MSB_pos, LSB_pos)

							# finding ceil(log of delta) and its corresponding weight
							log_delta_ceil = log_delta_floor + 1							
							ceil_weight = to_float_num(to_integer_num(last_unique_weight_log, MSB_pos, LSB_pos) + (2 ** log_delta_ceil), MSB_pos, LSB_pos)
							
							# finding which weight is closer to the real weight (floor or ceil) and assigning it to the kernel
							if((ceil_weight < abs_weights[i]) or (abs_weights[i] < floor_weight)):
								print("ERRROR!")
								print(str(to_integer_num(abs_weights[i] - last_unique_weight_log, MSB_pos, LSB_pos)))
								print("floor: " + str(floor_weight) + " floor log: " + str(log_delta_floor))
								print("weight: " + str(abs_weights[i]))
								print("last weight: " + str(last_unique_weight_log))
								print("ceil: " + str(ceil_weight) + " ceil log: " + str(log_delta_ceil))
								exit()

							if((ceil_weight - abs_weights[i]) < (abs_weights[i] - floor_weight)):
								last_unique_weight_log = ceil_weight
							else:
								last_unique_weight_log = floor_weight

							transposed_v2[T_o_c + o_c][T_i_c + i_c] = sign(transposed[T_o_c + o_c][T_i_c + i_c]) * last_unique_weight_log

						else:
							# using abs value for encoding weight
							num_of_abs_unique_weights_log = num_of_abs_unique_weights_log + 1
							transposed_v2[T_o_c + o_c][T_i_c + i_c] = quantization(transposed[T_o_c + o_c][T_i_c + i_c], MSB_pos, LSB_pos)
							last_unique_weight_log = abs_weights[i]
					
						# adding this new weight to our collection
						unique_weights_log.append(last_unique_weight_log)
						unique_weight_repetition_log.append(1)

					else:
						transposed_v2[T_o_c + o_c][T_i_c + i_c] = sign(transposed[T_o_c + o_c][T_i_c + i_c]) * last_unique_weight_log;
						if(len(unique_weight_repetition_log) == 0):
							unique_weight_repetition_log.append(1)
						else:
							unique_weight_repetition_log[-1] = unique_weight_repetition_log[-1] + 1



					# V2: using delta for encoding
					if(abs_weights[i] != last_unique_weight):
						# a new unique weight is found, encode it with either delta or abs approach

						if(to_integer_num(abs_weights[i] - last_unique_weight, MSB_pos, LSB_pos) <= (2 ** WEIGHT_DELTA_LEN[layer_name])):
							# using delta for encoding weight
							num_of_delta_unique_weights = num_of_delta_unique_weights + 1
						else:
							# using abs value for encoding weight
							num_of_abs_unique_weights = num_of_abs_unique_weights + 1

						# adding this new weight to our collection
						unique_weights.append(abs_weights[i])
						unique_weight_repetition.append(1)
						last_unique_weight = abs_weights[i]
					else:
						if(len(unique_weight_repetition) == 0):
							unique_weight_repetition.append(1)
						else:
							unique_weight_repetition[-1] = unique_weight_repetition[-1] + 1
					
					weight_file.write(str(quantization(transposed[T_o_c + o_c][T_i_c + i_c], MSB_pos, LSB_pos)) + " "),
					index_1d_file.write(str(index_1d[i]) + " "),
					index_ic_file.write(str(i_c) + " "),
				weight_file.write("\n"),
				index_1d_file.write("\n"),
				index_ic_file.write("\n"),
			weight_file.write("\n"),
			index_1d_file.write("\n"),
			index_ic_file.write("\n"),
	weight_file.close()
	index_1d_file.close()
	index_ic_file.close()
	for i in unique_weight_repetition:
		num_of_weight_repetition_cells += int((float(i - 1) / float((2 ** WEIGHT_NUM_LEN[layer_name]) - 2)) + 1);
	for i in unique_weight_repetition_log:
		num_of_weight_repetition_cells_log += int((float(i - 1) / float((2 ** WEIGHT_NUM_LEN_LOG[layer_name]) - 2)) + 1);


	writer.writerow([layer_name,
		"V1",
		o_c_num * i_c_num,
		num_of_zero_weights,
		len(unique_weights),
		num_of_abs_unique_weights,
		num_of_delta_unique_weights,
		num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN),
		num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name]),
		(num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])),
		num_of_weight_repetition_cells,
		num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name],
		(num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name])])
	print(layer_name + ": total V1: " + str((num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name])) + " WEIGHT_DELTA_LEN: " + str(WEIGHT_DELTA_LEN[layer_name]) + " WEIGHT_NUM_LEN: " + str(WEIGHT_NUM_LEN[layer_name]))
	# log_file.write(layer_name + ": total V1: " + str((num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name])) + " WEIGHT_DELTA_LEN: " + str(WEIGHT_DELTA_LEN[layer_name]) + " WEIGHT_NUM_LEN: " + str(WEIGHT_NUM_LEN[layer_name]))
	writer.writerow([layer_name,
		"V2",
		o_c_num * i_c_num,
		num_of_zero_weights,
		len(unique_weights_log),
		num_of_abs_unique_weights_log,
		num_of_delta_unique_weights_log,
		num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN),
		num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name]),
		(num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])),
		num_of_weight_repetition_cells_log,
		num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name],
		(num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name])])
	print(layer_name + ": total V2: " + str((num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name])) + " WEIGHT_DELTA_LEN_LOG: " + str(WEIGHT_DELTA_LEN_LOG[layer_name]) + " WEIGHT_NUM_LEN_LOG: " + str(WEIGHT_NUM_LEN_LOG[layer_name]))
	# log_file.write(layer_name + ": total V2: " + str((num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name])) + " WEIGHT_DELTA_LEN_LOG: " + str(WEIGHT_DELTA_LEN_LOG[layer_name]) + " WEIGHT_NUM_LEN_LOG: " + str(WEIGHT_NUM_LEN_LOG[layer_name]))

def cnn_layer_extractor(writer, net_data, net_data_v1, net_data_v2, layer_name):

	transposed = np.moveaxis(net_data[layer_name][0], [-1, -2], [0, 1])
	transposed_v1 = np.moveaxis(net_data_v1[layer_name][0], [-1, -2], [0, 1])
	transposed_v2 = np.moveaxis(net_data_v2[layer_name][0], [-1, -2], [0, 1])

	Max_weight = -1;
	Min_weight = 1;
	Max_abs_weight = 0;

	T_input_channel = 4
	T_output_channel = 4
	T_output_size = 8
	T_input_size_const = 40

	num_of_abs_unique_weights = 0;
	num_of_abs_unique_weights_log = 0;
	num_of_delta_unique_weights = 0;
	num_of_delta_unique_weights_log = 0;
	unique_weights = []
	unique_weights_log = []
	unique_weight_repetition = []
	unique_weight_repetition_log = []
	num_of_weight_repetition_cells = 0
	num_of_weight_repetition_cells_log = 0

	num_of_zero_weights = 0

	# for oc in net_data[layer_name][1]:
	# 	if(Max_weight < oc):
	# 		Max_weight = oc
	# 	if(Min_weight > oc):
	# 		Min_weight = oc
	# 	if(Max_abs_weight < abs(oc)):
	# 		Max_abs_weight = abs(oc)

	# for oc in transposed:
	# 	for ic in oc:
	# 		for kr in ic:
	# 			for kc in kr:
	# 				if(Max_weight < kc):
	# 					Max_weight = kc
	# 				if(Min_weight > kc):
	# 					Min_weight = kc
	# 				if(Max_abs_weight < abs(kc)):
	# 					Max_abs_weight = abs(kc)

	MSB_pos = 0 #int(log2(Max_abs_weight))
	LSB_pos = -7 #MSB_pos - (WEIGHT_ABS_LEN -1)

	# print("Max: " + str(Max_weight) + " Min: " + str(Min_weight) + " Max abs: " + str(Max_abs_weight) + " MSB position: " + str(MSB_pos) + " LSB position: " + str(LSB_pos))

	# bias manipulation
	for oc in range(len(net_data[layer_name][1])):
		net_data_v1[layer_name][1][oc] = quantization(net_data[layer_name][1][oc], MSB_pos, LSB_pos)
		net_data_v2[layer_name][1][oc] = quantization(net_data[layer_name][1][oc], MSB_pos, LSB_pos)

	weight_file = open(layer_name + "_weight_tiled.txt", "w")
	index_1d_file = open(layer_name + "_index_1d_tiled.txt", "w")
	index_oc_file = open(layer_name + "_index_oc_tiled.txt", "w")
	index_ic_file = open(layer_name + "_index_ic_tiled.txt", "w")
	index_kr_file = open(layer_name + "_index_kr_tiled.txt", "w")
	index_kc_file = open(layer_name + "_index_kc_tiled.txt", "w")
	o_c_num, i_c_num, k_r_num, k_c_num = shape(transposed)


	for T_o_c in range(0, o_c_num, T_output_channel):
		for T_i_c in range(0, i_c_num, T_input_channel):
			for i_c in range(0, T_input_channel):
				# for each input channel in tile, sort all weights corresponding to that channel and write output
				abs_weights = []
				index_1d = []
				if((T_i_c + i_c) >= i_c_num):
					continue
				for o_c in range(0, T_output_channel):
					if((T_o_c + o_c) >= o_c_num):
						continue
					for k_r in range(0, k_r_num):
						for k_c in range(0, k_c_num):
							transposed_v1[T_o_c + o_c][T_i_c + i_c][k_r][k_c] = quantization(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c], MSB_pos, LSB_pos)
							transposed_v2[T_o_c + o_c][T_i_c + i_c][k_r][k_c] = quantization(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c], MSB_pos, LSB_pos)
							if(quantization(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c], MSB_pos, LSB_pos) == 0):
								num_of_zero_weights = num_of_zero_weights + 1
								continue
							abs_weights.append(quantization(abs(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c]), MSB_pos, LSB_pos))
							index_1d.append(k_r*k_c_num*T_output_channel+k_c*T_output_channel+o_c)

				abs_weights, index_1d = merge_sort(abs_weights, index_1d)
				
				# sort indices related to unique weights
				unique_weight_indices = []
				first_unique_idx = 0
				for i in range(1, len(abs_weights) + 1):
					if(i == len(abs_weights)) or (abs_weights[i] != abs_weights[first_unique_idx]):
						index_1d[first_unique_idx:i] = merge_sort_single(index_1d[first_unique_idx:i])
						first_unique_idx = i

				last_unique_weight = 0;
				last_unique_weight_log = 0;

				for i in range(0, len(abs_weights)):
					k_r = int(index_1d[i] / (k_c_num * T_output_channel))
					k_c = int((index_1d[i] - (k_r * k_c_num * T_output_channel)) / T_output_channel)
					o_c = int(index_1d[i] - (k_r * k_c_num * T_output_channel) - (k_c * T_output_channel));

					# V1: using log(delta) for encoding
					if((abs_weights[i] > last_unique_weight_log) and (to_integer_num(abs_weights[i] - last_unique_weight_log, MSB_pos, LSB_pos) > 0)):
						# a new unique weight is found, encode it with either delta or abs approach

						if(log2(to_integer_num(abs_weights[i] - last_unique_weight_log, MSB_pos, LSB_pos)) < (2 ** WEIGHT_DELTA_LEN_LOG[layer_name])):
							# using log delta for encoding weight
							num_of_delta_unique_weights_log = num_of_delta_unique_weights_log + 1
							
							# finding floor(log of delta) and its corresponding weight
							log_delta_floor = int(log2(to_integer_num(abs_weights[i] - last_unique_weight_log, MSB_pos, LSB_pos)))
							floor_weight = to_float_num(to_integer_num(last_unique_weight_log, MSB_pos, LSB_pos) + (2 ** log_delta_floor), MSB_pos, LSB_pos)

							# finding ceil(log of delta) and its corresponding weight
							log_delta_ceil = log_delta_floor + 1							
							ceil_weight = to_float_num(to_integer_num(last_unique_weight_log, MSB_pos, LSB_pos) + (2 ** log_delta_ceil), MSB_pos, LSB_pos)
							
							# finding which weight is closer to the real weight (floor or ceil) and assigning it to the kernel
							if((ceil_weight < abs_weights[i]) or (abs_weights[i] < floor_weight)):
								print("ERRROR!")
								print(str(to_integer_num(abs_weights[i] - last_unique_weight_log, MSB_pos, LSB_pos)))
								print("floor: " + str(floor_weight) + " floor log: " + str(log_delta_floor))
								print("weight: " + str(abs_weights[i]))
								print("last weight: " + str(last_unique_weight_log))
								print("ceil: " + str(ceil_weight) + " ceil log: " + str(log_delta_ceil))
								exit()

							if((ceil_weight - abs_weights[i]) < (abs_weights[i] - floor_weight)):
								last_unique_weight_log = ceil_weight
							else:
								last_unique_weight_log = floor_weight

							transposed_v2[T_o_c + o_c][T_i_c + i_c][k_r][k_c] = sign(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c]) * last_unique_weight_log

						else:
							# using abs value for encoding weight
							num_of_abs_unique_weights_log = num_of_abs_unique_weights_log + 1
							transposed_v2[T_o_c + o_c][T_i_c + i_c][k_r][k_c] = quantization(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c], MSB_pos, LSB_pos)
							last_unique_weight_log = abs_weights[i]
					
						# adding this new weight to our collection
						unique_weights_log.append(last_unique_weight_log)
						unique_weight_repetition_log.append(1)

					else:
						transposed_v2[T_o_c + o_c][T_i_c + i_c][k_r][k_c] = sign(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c]) * last_unique_weight_log;
						if(len(unique_weight_repetition_log) == 0):
							unique_weight_repetition_log.append(1)
						else:
							unique_weight_repetition_log[-1] = unique_weight_repetition_log[-1] + 1



					# V2: using delta for encoding
					if(abs_weights[i] != last_unique_weight):
						# a new unique weight is found, encode it with either delta or abs approach

						if(to_integer_num(abs_weights[i] - last_unique_weight, MSB_pos, LSB_pos) <= (2 ** WEIGHT_DELTA_LEN[layer_name])):
							# using delta for encoding weight
							num_of_delta_unique_weights = num_of_delta_unique_weights + 1
						else:
							# using abs value for encoding weight
							num_of_abs_unique_weights = num_of_abs_unique_weights + 1

						# adding this new weight to our collection
						unique_weights.append(abs_weights[i])
						unique_weight_repetition.append(1)
						last_unique_weight = abs_weights[i]
					else:
						if(len(unique_weight_repetition) == 0):
							unique_weight_repetition.append(1)
						else:
							unique_weight_repetition[-1] = unique_weight_repetition[-1] + 1
					
					weight_file.write(str(quantization(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c], MSB_pos, LSB_pos)) + " "),
					index_1d_file.write(str(index_1d[i]) + " "),
					index_oc_file.write(str(o_c) + " "),
					index_ic_file.write(str(i_c) + " "),
					index_kr_file.write(str(k_r) + " "),
					index_kc_file.write(str(k_c) + " "),

				weight_file.write("\n"),
				index_1d_file.write("\n"),
				index_oc_file.write("\n"),
				index_ic_file.write("\n"),
				index_kr_file.write("\n"),
				index_kc_file.write("\n"),
			weight_file.write("\n"),
			index_1d_file.write("\n"),
			index_oc_file.write("\n"),
			index_ic_file.write("\n"),
			index_kr_file.write("\n"),
			index_kc_file.write("\n"),
	weight_file.close()
	index_1d_file.close()
	index_oc_file.close()
	index_ic_file.close()
	index_kr_file.close()
	index_kc_file.close()
	for i in unique_weight_repetition:
		num_of_weight_repetition_cells += int((float(i - 1) / float((2 ** WEIGHT_NUM_LEN[layer_name]) - 2)) + 1);
	for i in unique_weight_repetition_log:
		num_of_weight_repetition_cells_log += int((float(i - 1) / float((2 ** WEIGHT_NUM_LEN_LOG[layer_name]) - 2)) + 1);


	writer.writerow([layer_name,
		"V1",
		o_c_num * i_c_num * k_r_num * k_c_num,
		num_of_zero_weights,
		len(unique_weights),
		num_of_abs_unique_weights,
		num_of_delta_unique_weights,
		num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN),
		num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name]),
		(num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])),
		num_of_weight_repetition_cells,
		num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name],
		(num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name])])
	print(layer_name + ": total V1: " + str((num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name])) + " WEIGHT_DELTA_LEN: " + str(WEIGHT_DELTA_LEN[layer_name]) + " WEIGHT_NUM_LEN: " + str(WEIGHT_NUM_LEN[layer_name]))
	# log_file.write(layer_name + ": total V1: " + str((num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name])) + " WEIGHT_DELTA_LEN: " + str(WEIGHT_DELTA_LEN[layer_name]) + " WEIGHT_NUM_LEN: " + str(WEIGHT_NUM_LEN[layer_name]))
	writer.writerow([layer_name,
		"V2",
		o_c_num * i_c_num * k_r_num * k_c_num,
		num_of_zero_weights,
		len(unique_weights_log),
		num_of_abs_unique_weights_log,
		num_of_delta_unique_weights_log,
		num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN),
		num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name]),
		(num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])),
		num_of_weight_repetition_cells_log,
		num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name],
		(num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name])])
	print(layer_name + ": total V2: " + str((num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name])) + " WEIGHT_DELTA_LEN_LOG: " + str(WEIGHT_DELTA_LEN_LOG[layer_name]) + " WEIGHT_NUM_LEN_LOG: " + str(WEIGHT_NUM_LEN_LOG[layer_name]))
	# log_file.write(layer_name + ": total V2: " + str((num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name])) + " WEIGHT_DELTA_LEN_LOG: " + str(WEIGHT_DELTA_LEN_LOG[layer_name]) + " WEIGHT_NUM_LEN_LOG: " + str(WEIGHT_NUM_LEN_LOG[layer_name]))



net_data = load("bvlc_alexnet.npy", allow_pickle=True).item()
net_data_v1 = load("bvlc_alexnet.npy", allow_pickle=True).item()
net_data_v2 = load("bvlc_alexnet.npy", allow_pickle=True).item()


csv_file = open("alexnet_info.csv", 'w')
writer = csv.writer(csv_file)
writer.writerow(["Layer name",
	"version",
	"num of weights",
	"num of zero weights",
	"num of unique weights",
	"num of abs weights",
	"num of delta weights",
	"abs unique weights storage(bits) ",
	"delta uniqie weights storage(bits)",
	"total unique weights storage(bits)",
	"num of weight repetition elements",
	"unique weight repetition storage(bits)",
	"total weight storage(bits)"])


cnn_layer_extractor(writer, net_data, net_data_v1, net_data_v2, 'conv1')
cnn_layer_extractor(writer, net_data, net_data_v1, net_data_v2, 'conv2')
cnn_layer_extractor(writer, net_data, net_data_v1, net_data_v2, 'conv3')
cnn_layer_extractor(writer, net_data, net_data_v1, net_data_v2, 'conv4')
cnn_layer_extractor(writer, net_data, net_data_v1, net_data_v2, 'conv5')
fc_layer_extractor(writer, net_data, net_data_v1, net_data_v2, 'fc6')
fc_layer_extractor(writer, net_data, net_data_v1, net_data_v2, 'fc7')
fc_layer_extractor(writer, net_data, net_data_v1, net_data_v2, 'fc8')

np.save("bvlc_alexnet_v1.npy", net_data_v1)
np.save("bvlc_alexnet_v2.npy", net_data_v2)

print("finished")

exit()