#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IMAGE_ADDRESS "MNIST_dataset/t10k-images-idx3-ubyte"
#define LABEL_ADDRESS "MNIST_dataset/t10k-labels-idx1-ubyte"

#define CONV1_WEIGHT_ADDRESS "LeNet5_data/conv1_weights.txt"
#define CONV1_BIAS_ADDRESS "LeNet5_data/conv1_biasses.txt"
#define CONV2_WEIGHT_ADDRESS "LeNet5_data/conv2_weights.txt"
#define CONV2_BIAS_ADDRESS "LeNet5_data/conv2_biasses.txt"
#define CONV3_WEIGHT_ADDRESS "LeNet5_data/conv3_weights.txt"
#define CONV3_BIAS_ADDRESS "LeNet5_data/conv3_biasses.txt"
#define FC1_WEIGHT_ADDRESS "LeNet5_data/fc1_weights.txt"
#define FC1_BIAS_ADDRESS "LeNet5_data/fc1_biasses.txt"

#define MAX_TEST_SAMPLES 1000

#define MAX_CHANNEL 120
#define MAX_KERNEL_SIZE 5
#define MAX_FEATURE_SIZE 28

#define RELU_AF 0
#define TANH_AF 1
#define SELF_AF 2

#define MAX_POOL 0
#define MEAN_POOL 1


#define MAX_FEATURE_NUM 120

#define DEBUG_DEF 0
#define RESULT_SHOW 1

uint16_t lfsr = 0xACE1u;
unsigned period = 0;
char s[16+1];

FILE *fp_image;
int input_opened;

int random_gen(){
    unsigned lsb = lfsr & 1;

    lfsr >>= 1;

    if (lsb == 1)
        lfsr ^= 0xB400u;

    return lfsr;
}

int relu_af(int in_af){
	if(in_af < 0)
		return 0;

	return in_af;
}

void reset_cal(){
    input_opened = 0;
    if(fp_image != NULL)
        fclose(fp_image);
}

void read_cnn_inputs(unsigned char inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], unsigned char label[MAX_TEST_SAMPLES], int input_channel, int input_size, int num_of_samples, char* image_file, char* label_file){
    unsigned char temp;
    if(input_opened == 0){
        fp_image = fopen(image_file, "rb");
        FILE *fp_label = fopen(label_file, "rb");
        if (fp_image == NULL || fp_label == NULL){
            printf("ERROR. %s or %s doesn't exist\n", image_file, label_file);
            exit(-1);
        }
    	fseek(fp_image, 16, SEEK_SET);
    	fseek(fp_label, 8, SEEK_SET);
        for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
            for (int i_r_itr = 0; i_r_itr < input_size; i_r_itr++) {
                for (int i_c_itr = 0; i_c_itr < input_size; i_c_itr++) {
                    fread(&(inputs[i_ch_itr][i_r_itr][i_c_itr]), 1, 1, fp_image);
                }
            }
        }
        fread(label, num_of_samples, 1, fp_label);
    	fclose(fp_label);
        input_opened = 1;
    } else {
        for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
            for (int i_r_itr = 0; i_r_itr < input_size; i_r_itr++) {
                for (int i_c_itr = 0; i_c_itr < input_size; i_c_itr++) {
                    fread(&(inputs[i_ch_itr][i_r_itr][i_c_itr]), 1, 1, fp_image);
                }
            }
        }
    }
}

void read_cnn_weights(double weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE], double biasses[MAX_CHANNEL], int input_channel, int output_channel, int kernel_size, char* weight_file, char* bias_file){
    FILE *fp_weight = fopen(weight_file, "rb");
    FILE *fp_bias = fopen(bias_file, "rb");
    if (fp_weight == NULL || fp_bias == NULL){
        printf("ERROR. %s or %s doesn't exist\n", weight_file, bias_file);
        exit(-1);
    }
    for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
        fscanf(fp_bias, "%lf\n", &(biasses[o_ch_itr]));
        for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
            for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                    fscanf(fp_weight, "%lf", &(weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]));
                }
            }
        }
    }
	fclose(fp_weight);
	fclose(fp_bias);
}

void fc_layer(double weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM], double biasses[MAX_FEATURE_NUM], double inputs[MAX_FEATURE_NUM], double outputs[MAX_FEATURE_NUM], int input_num, int output_num, int af_num){
	for (int i = 0; i < output_num; i++) {
		outputs[i] = biasses[i];
		for (int j = 0; j < input_num; j++) {
			outputs[i] += weights[i][j] * inputs[j];
		}
        switch (af_num) {
            case RELU_AF:
                outputs[i] = (double)relu_af(outputs[i]);
                break;
            case TANH_AF:
                outputs[i] = (double)tanh(outputs[i]);
                break;
        }
	}
}

void fc_input_generator(double inputs[MAX_FEATURE_NUM], int input_num){
    for (int i = 0; i < input_num; i++) {
        inputs[i] = (random_gen() % 100) - 50;
    }
}

void fc_weight_generator(double weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM], double biasses[MAX_FEATURE_NUM], int input_num, int output_num){
    for (int i = 0; i < output_num; i++) {
        biasses[i] = (random_gen() % 10) - 5;
        for (int j = 0; j < input_num; j++) {
            weights[i][j] = (random_gen() % 100) - 50;
        }
    }
}

void read_fc_weights(double weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM], double biasses[MAX_FEATURE_NUM], int input_num, int output_num, char* weight_file, char* bias_file){
    FILE *fp_weight = fopen(weight_file, "rb");
    FILE *fp_bias = fopen(bias_file, "rb");
    if (fp_weight == NULL || fp_bias == NULL){
        printf("ERROR. %s or %s doesn't exist\n", weight_file, bias_file);
        exit(-1);
    }
    for (int i = 0; i < output_num; i++) {
        fscanf(fp_bias, "%lf\n", &(biasses[i]));
        for (int j = 0; j < input_num; j++) {
            fscanf(fp_weight, "%lf", &(weights[i][j]));
        }
    }
	fclose(fp_weight);
	fclose(fp_bias);
}

int fc_soft_max(double features[MAX_FEATURE_NUM], int feature_num){
    int max = 0;
    for (int i = 1; i < feature_num; i++) {
        if(features[max] < features[i])
            max = i;
    }
    return max;
}

void cnn_layer(double weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE], double biasses[MAX_CHANNEL], double inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], double outputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int input_channel, int output_channel, int input_size, int kernel_size, int stride, int zero_pad, int af_num){
    int output_size = ((input_size + (2 * zero_pad) - kernel_size) / stride) + 1;
    for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
        for (int o_r_itr = 0; o_r_itr < output_size; o_r_itr++) {
            for (int o_c_itr = 0; o_c_itr < output_size; o_c_itr++) {
                outputs[o_ch_itr][o_r_itr][o_c_itr] = biasses[o_ch_itr];
        		for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
                    for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                        for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                            outputs[o_ch_itr][o_r_itr][o_c_itr] += (((stride*o_r_itr)+k_r_itr-zero_pad) < 0) || (((stride*o_c_itr)+k_c_itr-zero_pad) < 0) || (((stride*o_r_itr)+k_r_itr-zero_pad) >= input_size) || (((stride*o_c_itr)+k_c_itr-zero_pad) >= input_size) ? 0 : inputs[i_ch_itr][(stride*o_r_itr)+k_r_itr-zero_pad][(stride*o_c_itr)+k_c_itr-zero_pad] * weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr];
                		}
                    }
        		}
                switch (af_num) {
                    case RELU_AF:
                        outputs[o_ch_itr][o_r_itr][o_c_itr] = (double)relu_af(outputs[o_ch_itr][o_r_itr][o_c_itr]);
                        break;
                    case TANH_AF:
                        outputs[o_ch_itr][o_r_itr][o_c_itr] = (double)tanh(outputs[o_ch_itr][o_r_itr][o_c_itr]);
                        break;
                }
    		}
		}
	}
}

void cnn_pool(double inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], double outputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int feature_channel, int input_size, int kernel_size, int stride, int zero_pad, int pool_num){
    int output_size = ((input_size + (2 * zero_pad) - kernel_size) / stride) + 1;
    double new_candidate;
    for (int ch_itr = 0; ch_itr < feature_channel; ch_itr++) {
        for (int o_r_itr = 0; o_r_itr < output_size; o_r_itr++) {
            for (int o_c_itr = 0; o_c_itr < output_size; o_c_itr++) {
                switch (pool_num) {
                    case MAX_POOL:
                        outputs[ch_itr][o_r_itr][o_c_itr] = inputs[ch_itr][stride*o_r_itr][stride*o_c_itr];
                        break;
                    case MEAN_POOL:
                        outputs[ch_itr][o_r_itr][o_c_itr] = 0.000000;
                        break;
                }
                for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                    for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                        new_candidate = inputs[ch_itr][(stride*o_r_itr)+k_r_itr][(stride*o_c_itr)+k_c_itr];
                        switch (pool_num) {
                            case MAX_POOL:
                                outputs[ch_itr][o_r_itr][o_c_itr] = (outputs[ch_itr][o_r_itr][o_c_itr] < new_candidate) ? new_candidate : outputs[ch_itr][o_r_itr][o_c_itr];
                                break;
                            case MEAN_POOL:
                                outputs[ch_itr][o_r_itr][o_c_itr] += new_candidate;
                                break;
                        }
                    }
        		}
                switch (pool_num) {
                    case MEAN_POOL:
                        outputs[ch_itr][o_r_itr][o_c_itr] /= (double)(kernel_size * kernel_size);
                        break;
                }
    		}
		}
	}
}

void cnn_input_generator(double inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int input_channel, int input_size){
    for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
        for (int i_r_itr = 0; i_r_itr < input_size; i_r_itr++) {
            for (int i_c_itr = 0; i_c_itr < input_size; i_c_itr++) {
                inputs[i_ch_itr][i_r_itr][i_c_itr] = (random_gen() % 100) - 50;
            }
        }
    }
}

void cnn_weight_generator(double weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE], double biasses[MAX_CHANNEL], int input_channel, int output_channel, int kernel_size){
    for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
        biasses[o_ch_itr] = (random_gen() % 10) - 5;
        for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
            for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                    weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr] = (random_gen() % 100) - 50;
                }
            }
        }
    }
}


#if(DEBUG_DEF == 1)
    void print_fc_weights(double weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM], int input_num, int output_num){
        for (int i = 0; i < output_num; i++) {
            printf("|\t");
            for (int j = 0; j < input_num; j++) {
                printf("%lf\t", weights[i][j]);
                if(j != input_num - 1){
                    printf(" ");
                }
            }
            printf("|\n");
        }
        printf("\n");
    }

    void print_fc_features(double feature[MAX_FEATURE_NUM], int feature_num){
        for (int i = 0; i < feature_num; i++) {
            printf("|\t%lf\t|\n", feature[i]);
        }
        printf("\n");
    }

    void print_cnn_weights(double weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE], int output_channel, int input_channel, int kernel_size){
        for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
            for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
                    printf("|\t");
                    for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                        printf("%lf\t", weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]);
                        if(k_c_itr != kernel_size - 1){
                            printf(" ");
                        }
                    }
                    printf("|\t");
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    void print_cnn_features(double feature[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int feature_channel, int feature_size){
        for (int ch_itr = 0; ch_itr < feature_channel; ch_itr++) {
            for (int r_itr = 0; r_itr < feature_size; r_itr++) {
                printf("|\t");
                for (int c_itr = 0; c_itr < feature_size; c_itr++) {
                    printf("%lf\t", feature[ch_itr][r_itr][c_itr]);
                }
                printf("|\n");
            }
            printf("\n");
        }
        printf("\n");
    }

#endif

void cnn_to_fc(double cnn_feature[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int cnn_feature_channel, int cnn_feature_size, double fc_feature[MAX_FEATURE_NUM]){
    for (int ch_itr = 0; ch_itr < cnn_feature_channel; ch_itr++) {
        for (int r_itr = 0; r_itr < cnn_feature_size; r_itr++) {
            for (int c_itr = 0; c_itr < cnn_feature_size; c_itr++) {
                fc_feature[(ch_itr*cnn_feature_size*cnn_feature_size)+(r_itr*cnn_feature_size)+c_itr] = cnn_feature[ch_itr][r_itr][c_itr];
            }
        }
    }
}

void LeNet(){

    double fc_weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM];
    double fc_biasses[MAX_FEATURE_NUM];
    double fc_inputs[MAX_FEATURE_NUM];
    double fc_outputs[MAX_FEATURE_NUM];
    int fc_input_num;
    int fc_output_num;

    unsigned char input_images[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE];
    unsigned char input_labels[MAX_TEST_SAMPLES];

    double cnn_weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    double cnn_biasses[MAX_CHANNEL];
    double cnn_inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE];
    double cnn_outputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE];
    int cnn_output_channel;
    int cnn_input_channel;
    int cnn_input_size;
    int cnn_output_size;
    int cnn_kernel_size;
    int cnn_stride;
    int cnn_zero_padd;
    int cnn_af_type;
    int cnn_pool_type;

    int result_index;
    int expected_index;

    int accuracy;
    int num_of_tests;

    accuracy = 0;
    num_of_tests = 1000;

    reset_cal();

    for (int itr = 0; itr < num_of_tests; itr++) {

        //layer 1
        cnn_output_channel = 6;
        cnn_input_channel = 1;
        cnn_input_size = 28;
        cnn_output_size = 28;
        cnn_kernel_size = 5;
        cnn_stride = 1;
        cnn_zero_padd = 2;
        cnn_af_type = RELU_AF;

        read_cnn_inputs(input_images, input_labels, cnn_input_channel, cnn_input_size, num_of_tests, IMAGE_ADDRESS, LABEL_ADDRESS);

        for (int a = 0; a < cnn_input_channel; a++) {
            for (int b = 0; b < cnn_input_size; b++) {
                for (int c = 0; c < cnn_input_size; c++) {
                    cnn_inputs[a][b][c] = (double)input_images[a][b][c];
                }
            }
        }

        read_cnn_weights(cnn_weights, cnn_biasses, cnn_input_channel, cnn_output_channel, cnn_kernel_size, CONV1_WEIGHT_ADDRESS, CONV1_BIAS_ADDRESS);
        cnn_layer(cnn_weights, cnn_biasses, cnn_inputs, cnn_outputs, cnn_input_channel, cnn_output_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_af_type);


#if(DEBUG_DEF == 1)
        printf("Layer 1: CNN:\n");
        printf("CNN weights:\n");
        print_cnn_weights(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size);
        printf("CNN Inputs:\n");
        print_cnn_features(cnn_inputs, cnn_input_channel, cnn_input_size);
        printf("CNN Outputs:\n");
        print_cnn_features(cnn_outputs, cnn_output_channel, cnn_output_size);
#endif

        cnn_output_channel = 6;
        cnn_input_channel = 6;
        cnn_input_size = 28;
        cnn_output_size = 14;
        cnn_kernel_size = 2;
        cnn_stride = 2;
        cnn_zero_padd = 0;
        cnn_pool_type = MAX_POOL;

        cnn_pool(cnn_outputs, cnn_inputs, cnn_input_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_pool_type);

#if(DEBUG_DEF == 1)
        printf("CNN Outputs:\n");
        print_cnn_features(cnn_inputs, cnn_output_channel, cnn_output_size);
#endif




        //layer 2
        cnn_output_channel = 16;
        cnn_input_channel = 6;
        cnn_input_size = 14;
        cnn_output_size = 10;
        cnn_kernel_size = 5;
        cnn_stride = 1;
        cnn_zero_padd = 0;
        cnn_af_type = RELU_AF;

        read_cnn_weights(cnn_weights, cnn_biasses, cnn_input_channel, cnn_output_channel, cnn_kernel_size, CONV2_WEIGHT_ADDRESS, CONV2_BIAS_ADDRESS);
        cnn_layer(cnn_weights, cnn_biasses, cnn_inputs, cnn_outputs, cnn_input_channel, cnn_output_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_af_type);


#if(DEBUG_DEF == 1)
        printf("Layer 2: CNN:\n");
        printf("CNN weights:\n");
        print_cnn_weights(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size);
        printf("CNN Inputs:\n");
        print_cnn_features(cnn_inputs, cnn_input_channel, cnn_input_size);
        printf("CNN Outputs:\n");
        print_cnn_features(cnn_outputs, cnn_output_channel, cnn_output_size);
#endif

        cnn_output_channel = 16;
        cnn_input_channel = 16;
        cnn_input_size = 10;
        cnn_output_size = 5;
        cnn_kernel_size = 2;
        cnn_stride = 2;
        cnn_zero_padd = 0;
        cnn_pool_type = MAX_POOL;

        cnn_pool(cnn_outputs, cnn_inputs, cnn_input_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_pool_type);

#if(DEBUG_DEF == 1)
        printf("CNN Outputs:\n");
        print_cnn_features(cnn_inputs, cnn_output_channel, cnn_output_size);
#endif

        //layer 3
        cnn_output_channel = 120;
        cnn_input_channel = 16;
        cnn_input_size = 5;
        cnn_output_size = 1;
        cnn_kernel_size = 5;
        cnn_stride = 1;
        cnn_zero_padd = 0;
        cnn_af_type = RELU_AF;

        read_cnn_weights(cnn_weights, cnn_biasses, cnn_input_channel, cnn_output_channel, cnn_kernel_size, CONV3_WEIGHT_ADDRESS, CONV3_BIAS_ADDRESS);
        cnn_layer(cnn_weights, cnn_biasses, cnn_inputs, cnn_outputs, cnn_input_channel, cnn_output_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_af_type);


#if(DEBUG_DEF == 1)
        printf("Layer 2: CNN:\n");
        printf("CNN weights:\n");
        print_cnn_weights(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size);
        printf("CNN Inputs:\n");
        print_cnn_features(cnn_inputs, cnn_input_channel, cnn_input_size);
        printf("CNN Outputs:\n");
        print_cnn_features(cnn_outputs, cnn_output_channel, cnn_output_size);
#endif


        //layer conversion
        cnn_to_fc(cnn_outputs, cnn_output_channel, cnn_output_size, fc_inputs);



        //first FC layer
        fc_input_num = 120;
        fc_output_num = 10;

        read_fc_weights(fc_weights, fc_biasses, fc_input_num, fc_output_num, FC1_WEIGHT_ADDRESS, FC1_BIAS_ADDRESS);
        fc_layer(fc_weights, fc_biasses, fc_inputs, fc_outputs, fc_input_num, fc_output_num, RELU_AF);

#if(DEBUG_DEF == 1)
        printf("Layer 3: FC:\n");
        printf("FC weights:\n");
        print_fc_weights(fc_weights, fc_input_num, fc_output_num);
        printf("FC Inputs:\n");
        print_fc_features(fc_inputs, fc_input_num);
        printf("FC Outputs:\n");
        print_fc_features(fc_outputs, fc_output_num);
#endif

        result_index = fc_soft_max(fc_outputs, fc_output_num);
        expected_index = input_labels[itr];

#if(RESULT_SHOW == 1)
        printf("itr: %d, expected: %d, result: %d\n", itr, expected_index, result_index);
#endif

        if(result_index == expected_index){
            accuracy += 1;
        }

    }

#if(RESULT_SHOW == 1)
    printf("accuracy: %d/%d\n", accuracy, num_of_tests);
#endif
    reset_cal();

}

int main(){

    LeNet();

    return 0;
}
