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

#define FRACTION_LEN 6
#define INTEGER_LEN 2
#define ESL_LEN 512

u_int16_t lfsr = 0xACE1u;
unsigned period = 0;
char s[16+1];

FILE *fp_image;
int input_opened;

typedef struct ESL_num {
    char X[ESL_LEN];
    char Y[ESL_LEN];
} ESL_num;

int random_generator(){
    unsigned lsb = lfsr & 1;

    lfsr >>= 1;

    if (lsb == 1)
        lfsr ^= 0xB400u;

    return lfsr;
}

// double relu_af(double in_af){
// 	if(in_af < 0.00000000)
// 		return 0.00000000;

// 	return in_af;
// }

// void reset_cal(){
//     input_opened = 0;
//     if(fp_image != NULL)
//         fclose(fp_image);
// }

// void read_cnn_inputs(unsigned char inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], unsigned char label[MAX_TEST_SAMPLES], int input_channel, int input_size, int num_of_samples, char* image_file, char* label_file){
//     unsigned char temp;
//     if(input_opened == 0){
//         fp_image = fopen(image_file, "rb");
//         FILE *fp_label = fopen(label_file, "rb");
//         if (fp_image == NULL || fp_label == NULL){
//             printf("ERROR. %s or %s doesn't exist\n", image_file, label_file);
//             exit(-1);
//         }
//     	fseek(fp_image, 16, SEEK_SET);
//     	fseek(fp_label, 8, SEEK_SET);
//         for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
//             for (int i_r_itr = 0; i_r_itr < input_size; i_r_itr++) {
//                 for (int i_c_itr = 0; i_c_itr < input_size; i_c_itr++) {
//                     fread(&(inputs[i_ch_itr][i_r_itr][i_c_itr]), 1, 1, fp_image);
//                 }
//             }
//         }
//         fread(label, num_of_samples, 1, fp_label);
//     	fclose(fp_label);
//         input_opened = 1;
//     } else {
//         for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
//             for (int i_r_itr = 0; i_r_itr < input_size; i_r_itr++) {
//                 for (int i_c_itr = 0; i_c_itr < input_size; i_c_itr++) {
//                     fread(&(inputs[i_ch_itr][i_r_itr][i_c_itr]), 1, 1, fp_image);
//                 }
//             }
//         }
//     }
// }

// double quantization(double input){
//     double sign;
//     sign = input < 0 ? -1.000000000 : 1.000000000;
//     input = input * sign;
//     return ((double)( (int)(input * pow(2, FRACTION_LEN)) & (int)(pow(2, FRACTION_LEN + INTEGER_LEN) - 1) ) / pow(2, FRACTION_LEN)) * sign;
// }

double SNG_to_double(ESL_num input){
    int num_of_ones_x = 0;
    int num_of_ones_y = 0;
    
    double in_x_double, in_y_double;

    for (int i = 0; i < ESL_LEN; ++i) {
        num_of_ones_x += input.X[i];
        num_of_ones_y += input.Y[i];
    }

    in_x_double = ((double)num_of_ones_x - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN);
    in_y_double = ((double)num_of_ones_y - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN);

    if(in_y_double == 0){
        printf("ERROR IN SNG_TO_DOUBLE, %d\n", num_of_ones_y);
        in_y_double = 0.001;
    }

    return in_x_double;// / in_y_double;

}


ESL_num multiplier(ESL_num first, ESL_num second) {
    ESL_num result;
    int num_of_ones = 0;

    for (int i = 0; i < ESL_LEN; ++i) {
        result.X[i] = (first.X[i] == second.X[i]) ? 1 : 0;
        result.Y[i] = (first.Y[i] == second.Y[i]) ? 1 : 0;
        num_of_ones += result.Y[i];
    }

    // printf("%lf\n", ((double)num_of_ones - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN));
    
    return result;

}

ESL_num adder(ESL_num first, ESL_num second) {
    ESL_num result;
    int y_num = 0;
    for (int i = 0; i < ESL_LEN; ++i) {
        result.X[i] = ((rand() % 2) == 0) ? ((first.X[i] == second.Y[i]) ? 1 : 0) : ((first.Y[i] == second.X[i]) ? 1 : 0);
        result.Y[i] = (((rand() % 2) + first.Y[i] + second.Y[i]) % 2 == 0) ? 1 : 0;
        y_num += result.Y[i];
    }
    if(y_num == 256){
        printf("%lf %lf\n", SNG_to_double(first), SNG_to_double(second));
    }
    return result;
}

ESL_num mac(ESL_num input, ESL_num weight, ESL_num initial){
    return adder(multiplier(input, weight), initial);
}

// void read_cnn_weights(double weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE], double biasses[MAX_CHANNEL], int input_channel, int output_channel, int kernel_size, char* weight_file, char* bias_file){
//     double temp;
    
//     FILE *fp_weight = fopen(weight_file, "rb");
//     FILE *fp_bias = fopen(bias_file, "rb");
//     if (fp_weight == NULL || fp_bias == NULL){
//         printf("ERROR. %s or %s doesn't exist\n", weight_file, bias_file);
//         exit(-1);
//     }
//     for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
//         fscanf(fp_bias, "%lf\n", &temp);
//         biasses[o_ch_itr] = quantization(temp);
//         for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
//             for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
//                 for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
//                     fscanf(fp_weight, "%lf", &temp);
//                     weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr] = quantization(temp);
//                 }
//             }
//         }
//     }
// 	fclose(fp_weight);
// 	fclose(fp_bias);
// }


ESL_num SNG(double input){
    int input_int = input * pow(2, FRACTION_LEN);
    int const_temp1 = pow(2, FRACTION_LEN + INTEGER_LEN);
    int const_temp2 = const_temp1 * 2;
    int num_of_ones = 0;
    ESL_num result;
    for (int i = 0; i < ESL_LEN; ++i) {
        result.X[i] = ((rand() % const_temp2) < (input_int + const_temp1)) ? 1 : 0;
        result.Y[i] = ((rand() % const_temp2) < (pow(2, FRACTION_LEN) + const_temp1)) ? 1 : 0;
        num_of_ones += result.Y[i];
    }

    // printf("Y = %lf\n", ((double)num_of_ones - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN));


    return result;
}

double bipolar_stochastic_divider(ESL_num input){
    int q_num_of_ones = 0;
    int approximate_num = pow(2, FRACTION_LEN + INTEGER_LEN);
    int sc_approximate_num;
    int approximate_p;
    int com_approximate;
    int max_approximate;
    int min_approximate;
    for (int i = 0; i < ESL_LEN; ++i) {
        q_num_of_ones += input.Y[i];
    }
    if(q_num_of_ones < pow(2, FRACTION_LEN + INTEGER_LEN)){
        for (int i = 0; i < ESL_LEN; ++i) {
            input.X[i] = 1 - input.X[i];
        }
    }
    for (int i = 0; i < ESL_LEN; ++i) {
        sc_approximate_num = ((rand() % (int)pow(2, FRACTION_LEN + INTEGER_LEN + 1)) < approximate_num) ? 1 : 0;
        approximate_p = (sc_approximate_num == input.X[i]) ? 1 : 0;
        com_approximate = (approximate_p != input.X[i]) ? 1 : 0;
        max_approximate = (com_approximate == (pow(2, FRACTION_LEN + INTEGER_LEN + 1) - 1)) ? 1 : 0;
        min_approximate = (com_approximate == 0) ? 1 : 0;
        if(com_approximate == 1){
            if(input.X[i] == 1 && max_approximate == 0){
                approximate_num += 1;
            } else if(input.X[i] == 0 && min_approximate == 0){
                approximate_num -= 1;
            }
        }
        // printf("%d-%lf ", approximate_num, ((double)(approximate_num - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN)));
    }
    return ((double)(approximate_num - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN));
}

// void fc_layer(double weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM], double biasses[MAX_FEATURE_NUM], double inputs[MAX_FEATURE_NUM], double outputs[MAX_FEATURE_NUM], int input_num, int output_num, int af_num){
// 	for (int i = 0; i < output_num; i++) {
// 		outputs[i] = biasses[i];
// 		for (int j = 0; j < input_num; j++) {
//             outputs[i] = mac(inputs[j], weights[i][j], outputs[i]);
// 		}

//         switch (af_num) {
//             case RELU_AF:
//                 outputs[i] = quantization((double)relu_af(outputs[i]));
//                 break;
//             case TANH_AF:
//                 outputs[i] = quantization((double)tanh(outputs[i]));
//                 break;
//         }
// 	}
// }

// void read_fc_weights(double weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM], double biasses[MAX_FEATURE_NUM], int input_num, int output_num, char* weight_file, char* bias_file){
//     double temp;
//     FILE *fp_weight = fopen(weight_file, "rb");
//     FILE *fp_bias = fopen(bias_file, "rb");
//     if (fp_weight == NULL || fp_bias == NULL){
//         printf("ERROR. %s or %s doesn't exist\n", weight_file, bias_file);
//         exit(-1);
//     }
//     for (int i = 0; i < output_num; i++) {
//         fscanf(fp_bias, "%lf\n", &temp);
//         biasses[i] = quantization(temp);
//         for (int j = 0; j < input_num; j++) {
//             fscanf(fp_weight, "%lf", &temp);
//             weights[i][j] = quantization(temp);
//         }
//     }
// 	fclose(fp_weight);
// 	fclose(fp_bias);
// }

// int fc_soft_max(double features[MAX_FEATURE_NUM], int feature_num){
//     int max = 0;
//     for (int i = 1; i < feature_num; i++) {
//         if(features[max] < features[i])
//             max = i;
//     }
//     return max;
// }

// void cnn_layer(double weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE], double biasses[MAX_CHANNEL], double inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], double outputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int input_channel, int output_channel, int input_size, int kernel_size, int stride, int zero_pad, int af_num){
//     int output_size = ((input_size + (2 * zero_pad) - kernel_size) / stride) + 1;
//     double temp;
//     for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
//         for (int o_r_itr = 0; o_r_itr < output_size; o_r_itr++) {
//             for (int o_c_itr = 0; o_c_itr < output_size; o_c_itr++) {
//                 outputs[o_ch_itr][o_r_itr][o_c_itr] = biasses[o_ch_itr];
//         		for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
//                     for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
//                         for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
//                             if((((stride*o_r_itr)+k_r_itr-zero_pad) < 0) || (((stride*o_c_itr)+k_c_itr-zero_pad) < 0) || (((stride*o_r_itr)+k_r_itr-zero_pad) >= input_size) || (((stride*o_c_itr)+k_c_itr-zero_pad) >= input_size)){
//                                 temp = 0.00000000;
//                             } else {
//                                 temp = quantization(inputs[i_ch_itr][(stride*o_r_itr)+k_r_itr-zero_pad][(stride*o_c_itr)+k_c_itr-zero_pad] * weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]);
//                             }
//                             outputs[o_ch_itr][o_r_itr][o_c_itr] = quantization(temp + outputs[o_ch_itr][o_r_itr][o_c_itr]);
//                 		}
//                     }
//         		}
//                 switch (af_num) {
//                     case RELU_AF:
//                         outputs[o_ch_itr][o_r_itr][o_c_itr] = quantization((double)relu_af(outputs[o_ch_itr][o_r_itr][o_c_itr]));
//                         break;
//                     case TANH_AF:
//                         outputs[o_ch_itr][o_r_itr][o_c_itr] = quantization((double)tanh(outputs[o_ch_itr][o_r_itr][o_c_itr]));
//                         break;
//                 }
//     		}
// 		}
// 	}
// }

// void cnn_pool(double inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], double outputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int feature_channel, int input_size, int kernel_size, int stride, int zero_pad, int pool_num){
//     int output_size = ((input_size + (2 * zero_pad) - kernel_size) / stride) + 1;
//     double new_candidate;
//     for (int ch_itr = 0; ch_itr < feature_channel; ch_itr++) {
//         for (int o_r_itr = 0; o_r_itr < output_size; o_r_itr++) {
//             for (int o_c_itr = 0; o_c_itr < output_size; o_c_itr++) {
//                 switch (pool_num) {
//                     case MAX_POOL:
//                         outputs[ch_itr][o_r_itr][o_c_itr] = inputs[ch_itr][stride*o_r_itr][stride*o_c_itr];
//                         break;
//                     case MEAN_POOL:
//                         outputs[ch_itr][o_r_itr][o_c_itr] = 0.000000;
//                         break;
//                 }
//                 for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
//                     for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
//                         new_candidate = inputs[ch_itr][(stride*o_r_itr)+k_r_itr][(stride*o_c_itr)+k_c_itr];
//                         switch (pool_num) {
//                             case MAX_POOL:
//                                 outputs[ch_itr][o_r_itr][o_c_itr] = (outputs[ch_itr][o_r_itr][o_c_itr] < new_candidate) ? new_candidate : outputs[ch_itr][o_r_itr][o_c_itr];
//                                 break;
//                             case MEAN_POOL:
//                                 outputs[ch_itr][o_r_itr][o_c_itr] = quantization(new_candidate + outputs[ch_itr][o_r_itr][o_c_itr]);
//                                 break;
//                         }
//                     }
//         		}
//                 switch (pool_num) {
//                     case MEAN_POOL:
//                         outputs[ch_itr][o_r_itr][o_c_itr] = quantization(outputs[ch_itr][o_r_itr][o_c_itr] / (double)(kernel_size * kernel_size));
//                         break;
//                 }
//     		}
// 		}
// 	}
// }


// #if(DEBUG_DEF == 1)
//     void print_fc_weights(double weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM], int input_num, int output_num){
//         for (int i = 0; i < output_num; i++) {
//             printf("|\t");
//             for (int j = 0; j < input_num; j++) {
//                 printf("%lf\t", weights[i][j]);
//                 if(j != input_num - 1){
//                     printf(" ");
//                 }
//             }
//             printf("|\n");
//         }
//         printf("\n");
//     }

//     void print_fc_features(double feature[MAX_FEATURE_NUM], int feature_num){
//         for (int i = 0; i < feature_num; i++) {
//             printf("|\t%lf\t|\n", feature[i]);
//         }
//         printf("\n");
//     }

//     void print_cnn_weights(double weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE], int output_channel, int input_channel, int kernel_size){
//         for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
//             for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
//                 for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
//                     printf("|\t");
//                     for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
//                         printf("%lf\t", weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]);
//                         if(k_c_itr != kernel_size - 1){
//                             printf(" ");
//                         }
//                     }
//                     printf("|\t");
//                 }
//                 printf("\n");
//             }
//             printf("\n");
//         }
//     }

//     void print_cnn_features(double feature[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int feature_channel, int feature_size){
//         for (int ch_itr = 0; ch_itr < feature_channel; ch_itr++) {
//             for (int r_itr = 0; r_itr < feature_size; r_itr++) {
//                 printf("|\t");
//                 for (int c_itr = 0; c_itr < feature_size; c_itr++) {
//                     printf("%lf\t", feature[ch_itr][r_itr][c_itr]);
//                 }
//                 printf("|\n");
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }

// #endif

// void cnn_to_fc(double cnn_feature[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE], int cnn_feature_channel, int cnn_feature_size, double fc_feature[MAX_FEATURE_NUM]){
//     for (int ch_itr = 0; ch_itr < cnn_feature_channel; ch_itr++) {
//         for (int r_itr = 0; r_itr < cnn_feature_size; r_itr++) {
//             for (int c_itr = 0; c_itr < cnn_feature_size; c_itr++) {
//                 fc_feature[(ch_itr*cnn_feature_size*cnn_feature_size)+(r_itr*cnn_feature_size)+c_itr] = cnn_feature[ch_itr][r_itr][c_itr];
//             }
//         }
//     }
// }

// void LeNet(){

//     double fc_weights[MAX_FEATURE_NUM][MAX_FEATURE_NUM];
//     double fc_biasses[MAX_FEATURE_NUM];
//     double fc_inputs[MAX_FEATURE_NUM];
//     double fc_outputs[MAX_FEATURE_NUM];
//     int fc_input_num;
//     int fc_output_num;

//     unsigned char input_images[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE];
//     unsigned char input_labels[MAX_TEST_SAMPLES];

//     double cnn_weights[MAX_CHANNEL][MAX_CHANNEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
//     double cnn_biasses[MAX_CHANNEL];
//     double cnn_inputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE];
//     double cnn_outputs[MAX_CHANNEL][MAX_FEATURE_SIZE][MAX_FEATURE_SIZE];
//     int cnn_output_channel;
//     int cnn_input_channel;
//     int cnn_input_size;
//     int cnn_output_size;
//     int cnn_kernel_size;
//     int cnn_stride;
//     int cnn_zero_padd;
//     int cnn_af_type;
//     int cnn_pool_type;

//     int result_index;
//     int expected_index;

//     int accuracy;
//     int num_of_tests;

//     accuracy = 0;
//     num_of_tests = 1000;

//     reset_cal();

//     for (int itr = 0; itr < num_of_tests; itr++) {

//         cnn_output_channel = 6;
//         cnn_input_channel = 1;
//         cnn_input_size = 28;
//         cnn_output_size = 28;
//         cnn_kernel_size = 5;
//         cnn_stride = 1;
//         cnn_zero_padd = 2;
//         cnn_af_type = RELU_AF;

//         read_cnn_inputs(input_images, input_labels, cnn_input_channel, cnn_input_size, num_of_tests, IMAGE_ADDRESS, LABEL_ADDRESS);

//         for (int a = 0; a < cnn_input_channel; a++) {
//             for (int b = 0; b < cnn_input_size; b++) {
//                 for (int c = 0; c < cnn_input_size; c++) {
//                     if(FRACTION_LEN < 8){
//                         cnn_inputs[a][b][c] = ((double)(input_images[a][b][c]/pow(2, 8 - FRACTION_LEN))/pow(2, FRACTION_LEN));
//                     } else {
//                         cnn_inputs[a][b][c] = ((double)input_images[a][b][c]/256.00000000);
//                     }
//                 }
//             }
//         }

//         read_cnn_weights(cnn_weights, cnn_biasses, cnn_input_channel, cnn_output_channel, cnn_kernel_size, CONV1_WEIGHT_ADDRESS, CONV1_BIAS_ADDRESS);
//         cnn_layer(cnn_weights, cnn_biasses, cnn_inputs, cnn_outputs, cnn_input_channel, cnn_output_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_af_type);


// #if(DEBUG_DEF == 1)
//         printf("Layer 1: CNN:\n");
//         printf("CNN weights:\n");
//         print_cnn_weights(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size);
//         printf("CNN Inputs:\n");
//         print_cnn_features(cnn_inputs, cnn_input_channel, cnn_input_size);
//         printf("CNN Outputs:\n");
//         print_cnn_features(cnn_outputs, cnn_output_channel, cnn_output_size);
// #endif

//         cnn_output_channel = 6;
//         cnn_input_channel = 6;
//         cnn_input_size = 28;
//         cnn_output_size = 14;
//         cnn_kernel_size = 2;
//         cnn_stride = 2;
//         cnn_zero_padd = 0;
//         cnn_pool_type = MAX_POOL;

//         cnn_pool(cnn_outputs, cnn_inputs, cnn_input_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_pool_type);

// #if(DEBUG_DEF == 1)
//         printf("CNN Outputs:\n");
//         print_cnn_features(cnn_inputs, cnn_output_channel, cnn_output_size);
// #endif




//         cnn_output_channel = 16;
//         cnn_input_channel = 6;
//         cnn_input_size = 14;
//         cnn_output_size = 10;
//         cnn_kernel_size = 5;
//         cnn_stride = 1;
//         cnn_zero_padd = 0;
//         cnn_af_type = RELU_AF;

//         read_cnn_weights(cnn_weights, cnn_biasses, cnn_input_channel, cnn_output_channel, cnn_kernel_size, CONV2_WEIGHT_ADDRESS, CONV2_BIAS_ADDRESS);
//         cnn_layer(cnn_weights, cnn_biasses, cnn_inputs, cnn_outputs, cnn_input_channel, cnn_output_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_af_type);


// #if(DEBUG_DEF == 1)
//         printf("Layer 2: CNN:\n");
//         printf("CNN weights:\n");
//         print_cnn_weights(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size);
//         printf("CNN Inputs:\n");
//         print_cnn_features(cnn_inputs, cnn_input_channel, cnn_input_size);
//         printf("CNN Outputs:\n");
//         print_cnn_features(cnn_outputs, cnn_output_channel, cnn_output_size);
// #endif

//         cnn_output_channel = 16;
//         cnn_input_channel = 16;
//         cnn_input_size = 10;
//         cnn_output_size = 5;
//         cnn_kernel_size = 2;
//         cnn_stride = 2;
//         cnn_zero_padd = 0;
//         cnn_pool_type = MAX_POOL;

//         cnn_pool(cnn_outputs, cnn_inputs, cnn_input_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_pool_type);

// #if(DEBUG_DEF == 1)
//         printf("CNN Outputs:\n");
//         print_cnn_features(cnn_inputs, cnn_output_channel, cnn_output_size);
// #endif

//         cnn_output_channel = 120;
//         cnn_input_channel = 16;
//         cnn_input_size = 5;
//         cnn_output_size = 1;
//         cnn_kernel_size = 5;
//         cnn_stride = 1;
//         cnn_zero_padd = 0;
//         cnn_af_type = RELU_AF;

//         read_cnn_weights(cnn_weights, cnn_biasses, cnn_input_channel, cnn_output_channel, cnn_kernel_size, CONV3_WEIGHT_ADDRESS, CONV3_BIAS_ADDRESS);
//         cnn_layer(cnn_weights, cnn_biasses, cnn_inputs, cnn_outputs, cnn_input_channel, cnn_output_channel, cnn_input_size, cnn_kernel_size, cnn_stride, cnn_zero_padd, cnn_af_type);


// #if(DEBUG_DEF == 1)
//         printf("Layer 2: CNN:\n");
//         printf("CNN weights:\n");
//         print_cnn_weights(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size);
//         printf("CNN Inputs:\n");
//         print_cnn_features(cnn_inputs, cnn_input_channel, cnn_input_size);
//         printf("CNN Outputs:\n");
//         print_cnn_features(cnn_outputs, cnn_output_channel, cnn_output_size);
// #endif


//         cnn_to_fc(cnn_outputs, cnn_output_channel, cnn_output_size, fc_inputs);



//         fc_input_num = 120;
//         fc_output_num = 10;

//         read_fc_weights(fc_weights, fc_biasses, fc_input_num, fc_output_num, FC1_WEIGHT_ADDRESS, FC1_BIAS_ADDRESS);
//         fc_layer(fc_weights, fc_biasses, fc_inputs, fc_outputs, fc_input_num, fc_output_num, RELU_AF);

// #if(DEBUG_DEF == 1)
//         printf("Layer 3: FC:\n");
//         printf("FC weights:\n");
//         print_fc_weights(fc_weights, fc_input_num, fc_output_num);
//         printf("FC Inputs:\n");
//         print_fc_features(fc_inputs, fc_input_num);
//         printf("FC Outputs:\n");
//         print_fc_features(fc_outputs, fc_output_num);
// #endif

//         result_index = fc_soft_max(fc_outputs, fc_output_num);
//         expected_index = input_labels[itr];

// #if(DEBUG_DEF == 1)
//         printf("itr: %d, expected: %d, result: %d\n", itr, expected_index, result_index);
// #endif

// #if(RESULT_SHOW == 1)
//         if(itr % 10 == 0){
//             printf("\r%d%% completed!", itr/10);
//             fflush(stdout);
//         }
// #endif

//         if(result_index == expected_index){
//             accuracy += 1;
//         }

//     }

// #if(RESULT_SHOW == 1)
//     printf("accuracy: %d/%d\n", accuracy, num_of_tests);
// #endif
//     reset_cal();

// }

double absolute(double input){
    if(input < 0.0000)
        return (-1.00000) * input;
    return input;
}

void mult_accuracy(){

    double num_double_1[1000];
    double num_double_2[1000];
    ESL_num num_ESL;
    ESL_num answer;

    double diff = 0;
    double curr_diff = 0;

    // printf("%lf == %lf\n", num_double_1[i], bipolar_stochastic_divider(SNG(num_double_1[i])));

    for (double j = 1.97; j > 0; j = j - 0.01){
        printf("%lf\n", j);
    }
    // return 0;
    printf("\n\n\n\n\ndiifs:\n\n\n\n");

    for (double j = 1.97; j > 0; j = j - 0.01){
        for (int i = 0; i < 1000; ++i){
            num_double_1[i] = ((double)(rand() & (int)(pow(2, FRACTION_LEN + INTEGER_LEN + 1) - 1)) - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN);
            num_double_2[i] = ((double)(rand() & (int)(pow(2, FRACTION_LEN + INTEGER_LEN + 1) - 1)) - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN);
            if(absolute(num_double_1[i]) <= j || absolute(num_double_2[i]) <= j || absolute(num_double_1[i] * num_double_2[i]) >= 4){
                i--;
                continue;
            }
            // printf("%lf\n", num_double_1[i] * num_double_2[i]);
        }

        diff = 0;
        for (int i = 0; i < 1000; ++i){
            answer = multiplier(SNG(num_double_1[i]), SNG(num_double_2[i]));
            curr_diff = absolute((absolute(num_double_1[i] * num_double_2[i]) - absolute(SNG_to_double(answer))) / (num_double_1[i] * num_double_2[i]));
            // printf("%lf * %lf = %lf, %lf, %lf\n", num_double_1[i], num_double_2[i], num_double_1[i] * num_double_2[i], 4.0000*SNG_to_double(answer), curr_diff);
            diff += curr_diff;
        }
        printf("%lf\n", diff/1000.00000);

    }

}

void one_sng_accuracy(){
    double num_double[1000];
    ESL_num answer;

    double diff = 0;
    double curr_diff = 0;

    diff = 0;
    for (int i = 0; i < 1000; ++i){
        answer = SNG(1.000000000);
        curr_diff = absolute(1.0000000 - SNG_to_double(answer));
        diff += curr_diff;
    }
    printf("%lf\n", diff/1000.00000);

}

void one_mult_accuracy(){
    ESL_num num_ESL;
    ESL_num answer;

    double diff = 0;
    double curr_diff = 0;

    for (int i = 0; i < 1000; ++i){
        answer = multiplier(SNG(1.00000), SNG(1.00000));
        curr_diff = absolute(1.000000 - (4.00000 * SNG_to_double(answer)));
        diff += curr_diff;
    }
    printf("%lf\n", diff/1000.00000);

}

void bipolar_divider_accuracy(){
    double num_double[1000];
    double answer;

    double diff = 0;
    double curr_diff = 0;

    for (double j = 3.99; j > -4; j = j - 0.01){
        printf("%lf\n", j);
    }
    // return 0;
    printf("\n\n\n\n\ndiifs:\n\n\n\n");

    for (double j = 3.99; j > -4; j = j - 0.01){

        diff = 0;
        for (int i = 0; i < 1000; ++i){
            answer = bipolar_stochastic_divider(SNG(j));
            curr_diff = absolute(j - answer) / absolute(j);
            diff += curr_diff;
        }
        printf("%lf\n", diff/1000.000000000);
    }
}

void add_accuracy(){

    double num_double_1[1000];
    double num_double_2[1000];
    ESL_num num_ESL;
    ESL_num answer;

    double diff = 0;
    double curr_diff = 0;

    // printf("%lf == %lf\n", num_double_1[i], bipolar_stochastic_divider(SNG(num_double_1[i])));

    for (double j = 1.97; j > 0; j = j - 0.01){
        printf("%lf\n", j);
    }
    // return 0;
    printf("\n\n\n\n\ndiifs:\n\n\n\n");

    for (double j = 1.97; j > 0; j = j - 0.01){
        for (int i = 0; i < 1000; ++i){
            num_double_1[i] = ((double)(rand() & (int)(pow(2, FRACTION_LEN + INTEGER_LEN + 1) - 1)) - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN);
            num_double_2[i] = ((double)(rand() & (int)(pow(2, FRACTION_LEN + INTEGER_LEN + 1) - 1)) - pow(2, FRACTION_LEN + INTEGER_LEN)) / pow(2, FRACTION_LEN);
            if(absolute(num_double_1[i]) <= j || absolute(num_double_2[i]) <= j || absolute(num_double_1[i] + num_double_2[i]) >= 4 || num_double_1[i] + num_double_2[i] == 0){
                i--;
                continue;
            }
            // printf("%lf\n", num_double_1[i] + num_double_2[i]);
        }

        diff = 0;
        for (int i = 0; i < 1000; ++i){
            answer = adder(SNG(num_double_1[i]), SNG(num_double_2[i]));
            printf("%lf + %lf = %lf-%lf\n", num_double_1[i], num_double_2[i], num_double_1[i] + num_double_2[i], SNG_to_double(answer));
            curr_diff = absolute((absolute(num_double_1[i] + num_double_2[i]) - absolute(SNG_to_double(answer))) / (num_double_1[i] + num_double_2[i]));
            diff += curr_diff;
        }
        printf("%lf\n", diff/1000.00000);

    }

}

int main(){
    add_accuracy();
    return 0;
}