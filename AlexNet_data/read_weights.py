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
IDX_DELTA_LEN = {
	"conv1" : 2,
	"conv2" : 1,
	"conv3" : 1,
	"conv4" : 1,
	"conv5" : 1,
	"fc6" : 3,
	"fc7" : 3,
	"fc8" : 4
}

T_input_channel = 4
T_output_channel = 4
T_output_size = 8
T_input_size_const = 40

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
        print(str(len(value)) + "\t" + str(len(i_1d)))

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

	

	T_input_num = T_output_size * T_output_size;
	T_output_num = 2 * T_output_size * T_output_size;

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

	modified_total_cycles = 0
	total_cycles = 0
	IDX_DELTA_LEN_SIZE = [0, 0, 0, 0, 0, 0, 0]


	MSB_pos = 0
	LSB_pos = -7

	# bias manipulation
	for oc in range(len(net_data[layer_name][1])):
		net_data_v1[layer_name][1][oc] = quantization(net_data[layer_name][1][oc], MSB_pos, LSB_pos)
		net_data_v2[layer_name][1][oc] = quantization(net_data[layer_name][1][oc], MSB_pos, LSB_pos)

	weight_file = open(layer_name + "_weight_tiled.txt", "w")
	index_1d_file = open(layer_name + "_index_1d_tiled.txt", "w")
	modified_index_1d_file = open(layer_name + "_index_1d_tiled_modified.txt", "w")

	for T_o_c in range(0, o_c_num, T_output_num):
		for T_i_c in range(0, i_c_num, T_input_num):
			indices = []
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

				indices.append([])

				for i in range(0, len(abs_weights)):
					indices[i_c].append(index_1d[i])
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
					
					weight_file.write(str(quantization(transposed[T_o_c + o_c][T_i_c + i_c], MSB_pos, LSB_pos)) + "\t"),

			# shows the idx in each line of indices that must be investigated
			indices_last_idx = []
			# shows if the line is finished
			indices_finished = []
			# indices version that has bubbles as well (bubble: -1)
			modified_indices = []
			# initializing both above arrays
			for i in range(0, len(indices)):
				modified_indices.append([])
				indices_last_idx.append(0)
				if(len(indices[i]) == 0):
					indices_finished.append(True)
				else:
					indices_finished.append(False)

			while True:
				# check if all lines are finished
				finished = True
				for i in range(0, len(indices)):
					if(indices_finished[i] == False):
						finished = False
						break
				if(finished):
					print("\r" + str(int(float(100 * (T_o_c * i_c_num + T_i_c)) / float(o_c_num * i_c_num))) + "% finished"),
					break

				# collect the oc of all unfinished lines
				oc_repetitions = []
				for i in range(0, T_output_num):
					oc_repetitions.append([])
				for i in range(0, len(indices)):
					# for each line
					if(indices_finished[i]):
						continue
					oc_repetitions[indices[i][indices_last_idx[i]]].append(i)

				# now we have collisions.
				for i in range(0, T_output_num):
					# for each output channel
					# print("i = " + str(i) + " repetition : " + str(oc_repetitions[i]))

					# if none of lines has this oc go to the next oc
					if(len(oc_repetitions[i]) == 0):
						continue

					# if only one line has this output channel schedule it
					line_number = oc_repetitions[i][0]
					if(len(oc_repetitions[i]) == 1):
						modified_indices[line_number].append(indices[line_number][indices_last_idx[line_number]])
						indices_last_idx[line_number] = indices_last_idx[line_number] + 1
						# if(indices_last_idx[line_number] == 37):
						# 	print("1")
						if(indices_last_idx[line_number] == len(indices[line_number])):
							indices_finished[line_number] = True
						continue

					# otherwise, more than one lines have this output channel
					# find the line that has more distance to end
					longest_line_distance = len(indices[line_number]) - indices_last_idx[line_number]
					longest_line_number = line_number
					for j in range(1, len(oc_repetitions[i])):
						line_number = oc_repetitions[i][j]
						if((len(indices[line_number]) - indices_last_idx[line_number]) > longest_line_distance):
							longest_line_distance = len(indices[line_number]) - indices_last_idx[line_number]
							longest_line_number = line_number
					# schedule that line

					# modified_indices[longest_line_number]
					# indices[longest_line_number]
					# print("line number = " + str(longest_line_number))
					# print("last idx = " + str(indices_last_idx[longest_line_number]))
					# print("indices = " + str(indices[longest_line_number]))

					modified_indices[longest_line_number].append(indices[longest_line_number][indices_last_idx[longest_line_number]])
					indices_last_idx[longest_line_number] = indices_last_idx[longest_line_number] + 1
					# if(indices_last_idx[longest_line_number] == 37):
					# 	print("2")
					if(indices_last_idx[longest_line_number] == len(indices[longest_line_number])):
						indices_finished[longest_line_number] = True
					# insert bubble to ther lines
					for j in range(0, len(oc_repetitions[i])):
						line_number = oc_repetitions[i][j]
						if(line_number == longest_line_number):
							continue
						modified_indices[line_number].append(-1)

			# performance evaluation
			max_line_len = 0
			modified_max_line_len = 0
			for line_number in range(0, len(modified_indices)):
				if(len(modified_indices[line_number]) > modified_max_line_len):
					modified_max_line_len = len(modified_indices[line_number])
				if(len(indices[line_number]) > max_line_len):
					max_line_len = len(indices[line_number])
			modified_total_cycles += modified_max_line_len
			total_cycles += max_line_len

			# encoded idx memory requirement evaluation
			for IDX_DELTA_LEN_itr in range(0, len(IDX_DELTA_LEN_SIZE)):
				for line_number in range(0, len(modified_indices)):
					# for each line
					last_idx = 0
					for idx_number in range(0, len(modified_indices[line_number])):
						# for each index in each line
						if(modified_indices[line_number][idx_number] == -1):
							# if it is bubble add size with 1
							IDX_DELTA_LEN_SIZE[IDX_DELTA_LEN_itr] += 1
						else:
							# if it is index
							larger_than_delta = (modified_indices[line_number][idx_number] - last_idx) > (2 ** (IDX_DELTA_LEN_itr + 1))
							new_weight = last_idx > modified_indices[line_number][idx_number]
							if larger_than_delta or new_weight:
								# idx is absolute
								IDX_DELTA_LEN_SIZE[IDX_DELTA_LEN_itr] += (2 + log2(T_output_num))
							else:
								# idx is delta
								IDX_DELTA_LEN_SIZE[IDX_DELTA_LEN_itr] += (2 + IDX_DELTA_LEN_itr + 1)
							last_idx = modified_indices[line_number][idx_number]

			# writing indices
			for line_number in range(0, len(indices)):
				# for each line
				for idx_number in range(0, len(indices[line_number])):
					# for each index in each line
					index_1d_file.write(str(indices[line_number][idx_number]) + "\t"),
				index_1d_file.write("\n"),

			# writing modified indices
			for line_number in range(0, len(modified_indices)):
				# for each line
				for idx_number in range(0, len(modified_indices[line_number])):
					# for each index in each line
					modified_index_1d_file.write(str(modified_indices[line_number][idx_number]) + "\t"),
				modified_index_1d_file.write("\n"),

			weight_file.write("\n"),
			index_1d_file.write("\n"),
			modified_index_1d_file.write("\n"),

	weight_file.close()
	index_1d_file.close()
	modified_index_1d_file.close()
	for i in unique_weight_repetition:
		num_of_weight_repetition_cells += int((float(i - 1) / float((2 ** WEIGHT_NUM_LEN[layer_name]) - 2)) + 1);
	for i in unique_weight_repetition_log:
		num_of_weight_repetition_cells_log += int((float(i - 1) / float((2 ** WEIGHT_NUM_LEN_LOG[layer_name]) - 2)) + 1);

	min_memory_requirement = 100000000000
	min_memory_requirement_param = 0
	for i in range(0, len(IDX_DELTA_LEN_SIZE)):
		if(IDX_DELTA_LEN_SIZE[i] < min_memory_requirement):
			min_memory_requirement = IDX_DELTA_LEN_SIZE[i]
			min_memory_requirement_param = i + 1

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
		(num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name]),
		total_cycles,
		modified_total_cycles,
		100.0000 * float(modified_total_cycles - total_cycles) / float(total_cycles),
		min_memory_requirement,
		min_memory_requirement_param])
	print(layer_name + ": total V1: " + str((num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name])) + " WEIGHT_DELTA_LEN: " + str(WEIGHT_DELTA_LEN[layer_name]) + " WEIGHT_NUM_LEN: " + str(WEIGHT_NUM_LEN[layer_name]))
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
		(num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name]),
		total_cycles,
		modified_total_cycles,
		100.0000 * float(modified_total_cycles - total_cycles) / float(total_cycles),
		min_memory_requirement,
		min_memory_requirement_param])
	print(layer_name + ": total V2: " + str((num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name])) + " WEIGHT_DELTA_LEN_LOG: " + str(WEIGHT_DELTA_LEN_LOG[layer_name]) + " WEIGHT_NUM_LEN_LOG: " + str(WEIGHT_NUM_LEN_LOG[layer_name]))


def cnn_layer_extractor(writer, net_data, net_data_v1, net_data_v2, layer_name):

	transposed = np.moveaxis(net_data[layer_name][0], [-1, -2], [0, 1])
	transposed_v1 = np.moveaxis(net_data_v1[layer_name][0], [-1, -2], [0, 1])
	transposed_v2 = np.moveaxis(net_data_v2[layer_name][0], [-1, -2], [0, 1])

	Max_weight = -1;
	Min_weight = 1;
	Max_abs_weight = 0;

	

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

	modified_total_cycles = 0
	total_cycles = 0
	IDX_DELTA_LEN_SIZE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	MSB_pos = 0
	LSB_pos = -7

	# bias manipulation
	for oc in range(len(net_data[layer_name][1])):
		net_data_v1[layer_name][1][oc] = quantization(net_data[layer_name][1][oc], MSB_pos, LSB_pos)
		net_data_v2[layer_name][1][oc] = quantization(net_data[layer_name][1][oc], MSB_pos, LSB_pos)

	weight_file = open(layer_name + "_weight_tiled.txt", "w")
	index_1d_file = open(layer_name + "_index_1d_tiled.txt", "w")
	modified_index_1d_file = open(layer_name + "_index_1d_tiled_modified.txt", "w")
	index_oc_file = open(layer_name + "_index_oc_tiled.txt", "w")
	modified_index_oc_file = open(layer_name + "_index_oc_tiled_modified.txt", "w")
	index_ic_file = open(layer_name + "_index_ic_tiled.txt", "w")
	index_kr_file = open(layer_name + "_index_kr_tiled.txt", "w")
	index_kc_file = open(layer_name + "_index_kc_tiled.txt", "w")
	o_c_num, i_c_num, k_r_num, k_c_num = shape(transposed)


	for T_o_c in range(0, o_c_num, T_output_channel):
		for T_i_c in range(0, i_c_num, T_input_channel):
			indices = []
			ocs = []
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

				indices.append([])
				ocs.append([])

				for i in range(0, len(abs_weights)):

					k_r = int(index_1d[i] / (k_c_num * T_output_channel))
					k_c = int((index_1d[i] - (k_r * k_c_num * T_output_channel)) / T_output_channel)
					o_c = int(index_1d[i] - (k_r * k_c_num * T_output_channel) - (k_c * T_output_channel));

					indices[i_c].append(index_1d[i])
					ocs[i_c].append(o_c)

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
					
					weight_file.write(str(quantization(transposed[T_o_c + o_c][T_i_c + i_c][k_r][k_c], MSB_pos, LSB_pos)) + "\t"),
					index_ic_file.write(str(i_c) + "\t"),
					index_kr_file.write(str(k_r) + "\t"),
					index_kc_file.write(str(k_c) + "\t"),

				weight_file.write("\n"),
				index_ic_file.write("\n"),
				index_kr_file.write("\n"),
				index_kc_file.write("\n"),

			# shows the idx in each line of indices that must be investigated
			indices_last_idx = []
			# shows if the line is finished
			indices_finished = []
			# indices version that has bubbles as well (bubble: -1)
			modified_indices = []
			# oc version that has bubbles as well (bubble: -1)
			modified_ocs = []
			# initializing both above arrays
			for i in range(0, len(indices)):
				modified_indices.append([])
				modified_ocs.append([])
				indices_last_idx.append(0)
				if(len(indices[i]) == 0):
					indices_finished.append(True)
				else:
					indices_finished.append(False)

			while True:
				# check if all lines are finished
				finished = True
				for i in range(0, len(indices)):
					if(indices_finished[i] == False):
						finished = False
						break
				if(finished):
					print("\r" + str(int(float(100 * (T_o_c * i_c_num + T_i_c)) / float(o_c_num * i_c_num))) + "% finished"),
					break

				# collect the oc of all unfinished lines
				oc_repetitions = []
				for i in range(0, T_output_channel):
					oc_repetitions.append([])
				for i in range(0, len(indices)):
					# for each line
					if(indices_finished[i]):
						continue
					oc_repetitions[ocs[i][indices_last_idx[i]]].append(i)

				# now we have collisions.
				for i in range(0, T_output_channel):
					# for each output channel

					# if none of lines has this oc go to the next oc
					if(len(oc_repetitions[i]) == 0):
						continue

					# if only one line has this output channel schedule it
					line_number = oc_repetitions[i][0]
					if(len(oc_repetitions[i]) == 1):
						modified_indices[line_number].append(indices[line_number][indices_last_idx[line_number]])
						modified_ocs[line_number].append(ocs[line_number][indices_last_idx[line_number]])
						indices_last_idx[line_number] = indices_last_idx[line_number] + 1
						if(indices_last_idx[line_number] == len(indices[line_number])):
							indices_finished[line_number] = True
						continue

					# otherwise, more than one lines have this output channel
					# find the line that has more distance to end
					longest_line_distance = len(indices[line_number]) - indices_last_idx[line_number]
					longest_line_number = line_number
					for j in range(1, len(oc_repetitions[i])):
						line_number = oc_repetitions[i][j]
						if((len(indices[line_number]) - indices_last_idx[line_number]) > longest_line_distance):
							longest_line_distance = len(indices[line_number]) - indices_last_idx[line_number]
							longest_line_number = line_number
					
					# schedule that line
					modified_indices[longest_line_number].append(indices[longest_line_number][indices_last_idx[longest_line_number]])
					modified_ocs[longest_line_number].append(ocs[longest_line_number][indices_last_idx[longest_line_number]])
					indices_last_idx[longest_line_number] = indices_last_idx[longest_line_number] + 1
					if(indices_last_idx[longest_line_number] == len(indices[longest_line_number])):
						indices_finished[longest_line_number] = True
					# insert bubble to ther lines
					for j in range(0, len(oc_repetitions[i])):
						line_number = oc_repetitions[i][j]
						if(line_number == longest_line_number):
							continue
						modified_indices[line_number].append(-1)
						modified_ocs[line_number].append(-1)

			# performance evaluation
			max_line_len = 0
			modified_max_line_len = 0
			for line_number in range(0, len(modified_indices)):
				if(len(modified_indices[line_number]) > modified_max_line_len):
					modified_max_line_len = len(modified_indices[line_number])
				if(len(indices[line_number]) > max_line_len):
					max_line_len = len(indices[line_number])
			modified_total_cycles += modified_max_line_len
			total_cycles += max_line_len

			# encoded idx memory requirement evaluation
			for IDX_DELTA_LEN_itr in range(0, len(IDX_DELTA_LEN_SIZE)):
				for line_number in range(0, len(modified_indices)):
					# for each line
					last_idx = 0
					for idx_number in range(0, len(modified_indices[line_number])):
						# for each index in each line
						if(modified_indices[line_number][idx_number] == -1):
							# if it is bubble add size with 1
							IDX_DELTA_LEN_SIZE[IDX_DELTA_LEN_itr] += 1
						else:
							# if it is index
							larger_than_delta = (modified_indices[line_number][idx_number] - last_idx) > (2 ** (IDX_DELTA_LEN_itr + 1))
							new_weight = last_idx > modified_indices[line_number][idx_number]
							if larger_than_delta or new_weight:
								# idx is absolute
								IDX_DELTA_LEN_SIZE[IDX_DELTA_LEN_itr] += (2 + log2(T_output_channel) + log2(k_r_num) + log2(k_c_num))
							else:
								# idx is delta
								IDX_DELTA_LEN_SIZE[IDX_DELTA_LEN_itr] += (2 + IDX_DELTA_LEN_itr + 1)
							last_idx = modified_indices[line_number][idx_number]

			# writing indices
			for line_number in range(0, len(indices)):
				# for each line
				for idx_number in range(0, len(indices[line_number])):
					# for each index in each line
					index_1d_file.write(str(indices[line_number][idx_number]) + "\t"),
				index_1d_file.write("\n"),

			# writing modified indices
			for line_number in range(0, len(modified_indices)):
				# for each line
				for idx_number in range(0, len(modified_indices[line_number])):
					# for each index in each line
					modified_index_1d_file.write(str(modified_indices[line_number][idx_number]) + "\t"),
				modified_index_1d_file.write("\n"),

			# writing ocs
			for line_number in range(0, len(ocs)):
				# for each line
				for idx_number in range(0, len(ocs[line_number])):
					# for each index in each line
					index_oc_file.write(str(ocs[line_number][idx_number]) + "\t"),
				index_oc_file.write("\n"),

			# writing modified ocs
			for line_number in range(0, len(modified_ocs)):
				# for each line
				for idx_number in range(0, len(modified_ocs[line_number])):
					# for each index in each line
					modified_index_oc_file.write(str(modified_ocs[line_number][idx_number]) + "\t"),
				modified_index_oc_file.write("\n"),

			weight_file.write("\n"),
			index_1d_file.write("\n"),
			index_oc_file.write("\n"),
			modified_index_1d_file.write("\n"),
			modified_index_oc_file.write("\n"),
			index_ic_file.write("\n"),
			index_kr_file.write("\n"),
			index_kc_file.write("\n"),
	weight_file.close()
	index_1d_file.close()
	index_oc_file.close()
	modified_index_1d_file.close()
	modified_index_oc_file.close()
	index_ic_file.close()
	index_kr_file.close()
	index_kc_file.close()
	for i in unique_weight_repetition:
		num_of_weight_repetition_cells += int((float(i - 1) / float((2 ** WEIGHT_NUM_LEN[layer_name]) - 2)) + 1);
	for i in unique_weight_repetition_log:
		num_of_weight_repetition_cells_log += int((float(i - 1) / float((2 ** WEIGHT_NUM_LEN_LOG[layer_name]) - 2)) + 1);

	min_memory_requirement = 100000000000
	min_memory_requirement_param = 0
	for i in range(0, len(IDX_DELTA_LEN_SIZE)):
		if(IDX_DELTA_LEN_SIZE[i] < min_memory_requirement):
			min_memory_requirement = IDX_DELTA_LEN_SIZE[i]
			min_memory_requirement_param = i + 1

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
		(num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name]),
		total_cycles,
		modified_total_cycles,
		100.0000 * float(modified_total_cycles - total_cycles) / float(total_cycles),
		min_memory_requirement,
		min_memory_requirement_param])
	print(layer_name + ": total V1: " + str((num_of_abs_unique_weights * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights * (1 + WEIGHT_DELTA_LEN[layer_name])) + (num_of_weight_repetition_cells * WEIGHT_NUM_LEN[layer_name])) + " WEIGHT_DELTA_LEN: " + str(WEIGHT_DELTA_LEN[layer_name]) + " WEIGHT_NUM_LEN: " + str(WEIGHT_NUM_LEN[layer_name]))
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
		(num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name]),
		total_cycles,
		modified_total_cycles,
		100.0000 * float(modified_total_cycles - total_cycles) / float(total_cycles),
		min_memory_requirement,
		min_memory_requirement_param])
	print(layer_name + ": total V2: " + str((num_of_abs_unique_weights_log * (1 + WEIGHT_ABS_LEN)) + (num_of_delta_unique_weights_log * (1 + WEIGHT_DELTA_LEN_LOG[layer_name])) + (num_of_weight_repetition_cells_log * WEIGHT_NUM_LEN_LOG[layer_name])) + " WEIGHT_DELTA_LEN_LOG: " + str(WEIGHT_DELTA_LEN_LOG[layer_name]) + " WEIGHT_NUM_LEN_LOG: " + str(WEIGHT_NUM_LEN_LOG[layer_name]))



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
	"total weight storage(bits)",
	"total cycle without bubbles",
	"total cycle with bubbles",
	"total cycle overhead",
	"min total idx memory storage",
	"IDX_DELTA_LEN"])

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