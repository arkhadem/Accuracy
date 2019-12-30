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
#define DEBUG_DEF_FILE 1
#define RESULT_SHOW 1

#define BIN_LEN 12
#define FRACTION_LEN 9
#define INTEGER_LEN 2
#define ESL_LEN 4096

u_int16_t lfsr = 0xACE1u;
unsigned period = 0;
char s[16+1];

FILE *fp_image;
int input_opened;

typedef struct ESL_num {
    char X[ESL_LEN];
    char Y[ESL_LEN];
} ESL_num;

double absolute(double input);
double quantization(double input);
int random_generator();
ESL_num SNG(double input);
ESL_num relu_af(ESL_num input);
ESL_num tanh_af(ESL_num input);
void reset_cal();
void read_cnn_inputs(ESL_num*** inputs, unsigned char label[MAX_TEST_SAMPLES], int input_channel, int input_size, int num_of_samples, char* image_file, char* label_file);
double ESL_to_double_nominator(ESL_num input, char type);
double ESL_to_double(ESL_num input);
ESL_num multiplier(ESL_num first, ESL_num second) ;
ESL_num adder_arr(ESL_num* input, int num_of_elements);
ESL_num adder_arr_1(ESL_num* input, int num_of_elements);
ESL_num adder_arr_2(ESL_num* input, int num_of_elements);
ESL_num adder(ESL_num first, ESL_num second) ;
ESL_num adder_2(ESL_num first, ESL_num second) ;
ESL_num mac(ESL_num input, ESL_num weight, ESL_num initial);
void read_cnn_weights(ESL_num**** weights, ESL_num* biasses, int input_channel, int output_channel, int kernel_size, char* weight_file, char* bias_file);
double bipolar_stochastic_divider(ESL_num input);
void fc_layer(ESL_num** weights, ESL_num* biasses, ESL_num* inputs, ESL_num* outputs, int input_num, int output_num, int af_num);
void read_fc_weights(ESL_num** weights, ESL_num* biasses, int input_num, int output_num, char* weight_file, char* bias_file);
int fc_soft_max(ESL_num inputs[MAX_FEATURE_NUM], int feature_num);
void cnn_layer(ESL_num**** weights, ESL_num* biasses, ESL_num*** inputs, ESL_num*** outputs, int input_channel, int output_channel, int input_size, int kernel_size, int stride, int zero_pad, int af_num);
int A_l_B(ESL_num first, ESL_num second);
void cnn_pool(ESL_num*** inputs, ESL_num*** outputs, int feature_channel, int input_size, int kernel_size, int stride, int zero_pad, int pool_num);
void cnn_to_fc(ESL_num*** cnn_feature, int cnn_feature_channel, int cnn_feature_size, ESL_num* fc_feature);
void print_cnn_features(ESL_num*** feature, int feature_channel, int feature_size);

double absolute(double input){
    if(input < 0.0000)
        return (-1.00000) * input;
    return input;
}

double quantization(double input){
    double sign;
    sign = input < 0 ? -1.000000000 : 1.000000000;
    input = input * sign;
    return ((double)( (int)(input * pow(2, FRACTION_LEN)) & (int)(pow(2, FRACTION_LEN + INTEGER_LEN) - 1) ) / pow(2, FRACTION_LEN)) * sign;
}

int random_generator(){
    unsigned lsb = lfsr & 1;

    lfsr >>= 1;

    if (lsb == 1)
        lfsr ^= 0xB400u;

    return lfsr;
}
ESL_num SNG(double input){
    int max_num = pow(2, BIN_LEN);
    int num_of_ones = 0;
    ESL_num result;
    double nominator, denominator;

    if(absolute(input) >= 1){
        nominator = 1.000000;
        denominator = 1.000000 / input;
    } else{
        nominator = input;
        denominator = 1.000000;
    }

    int nominator_int = (int)((nominator + 1) * pow(2, BIN_LEN - 1));
    int denominator_int = (int)((denominator + 1) * pow(2, BIN_LEN - 1));

    for (int i = 0; i < ESL_LEN; ++i) {
        result.X[i] = ((rand() % max_num) < nominator_int) ? 1 : 0;
        result.Y[i] = ((rand() % max_num) < denominator_int) ? 1 : 0;
    }

    return result;
}

ESL_num relu_af(ESL_num input){
    double in_af = ESL_to_double(input);
	if(in_af < 0.00000000)
		return SNG(0.00000000);
    else
        return input;
}

ESL_num tanh_af(ESL_num input){
    double in_af = ESL_to_double(input);
    return SNG(tanh(in_af));
}

void reset_cal(){
    input_opened = 0;
    if(fp_image != NULL)
        fclose(fp_image);
}

void read_cnn_inputs(ESL_num*** inputs, unsigned char label[MAX_TEST_SAMPLES], int input_channel, int input_size, int num_of_samples, char* image_file, char* label_file){
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
                    fread(&temp, 1, 1, fp_image);
                    if(FRACTION_LEN < 8){
                        inputs[i_ch_itr][i_r_itr][i_c_itr] = SNG((double)(temp/pow(2, 8)));
                        //printf("%lf\n", ESL_to_double(inputs[i_ch_itr][i_r_itr][i_c_itr]));
                    } else {
                        inputs[i_ch_itr][i_r_itr][i_c_itr] = SNG((double)temp/256.00000000);
                        //printf("%lf\n", ESL_to_double(inputs[i_ch_itr][i_r_itr][i_c_itr]));
                    }
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
                    fread(&temp, 1, 1, fp_image);
                    if(FRACTION_LEN < 8){
                        inputs[i_ch_itr][i_r_itr][i_c_itr] = SNG((double)(temp/pow(2, 8 - FRACTION_LEN))/pow(2, FRACTION_LEN));
                    } else {
                        inputs[i_ch_itr][i_r_itr][i_c_itr] = SNG((double)temp/256.00000000);
                    }
                }
            }
        }
    }
    //print_cnn_features(inputs, input_channel, input_size);
}

double ESL_to_double_nominator(ESL_num input, char type){
    int num_of_ones = 0;
    for (int i = 0; i < ESL_LEN; i++) {
        num_of_ones += (type == 0) ? input.Y[i] : input.X[i];
    }
    return ((double)(num_of_ones)/pow(2, BIN_LEN - 1)) - 1.00000;
}

double ESL_to_double(ESL_num input){
    double in_x_double = ESL_to_double_nominator(input, 1);
    double in_y_double = ESL_to_double_nominator(input, 0);

    if(in_y_double == 0){
        in_y_double = 0.1;
    }

    return in_x_double / in_y_double;

}

ESL_num multiplier(ESL_num first, ESL_num second) {
    //return SNG(ESL_to_double(first)*ESL_to_double(second));
    ESL_num result;
    int num_of_ones_x = 0;
    int num_of_ones_y = 0;

    for (int i = 0; i < ESL_LEN; ++i) {
        result.X[i] = (first.X[i] == second.X[i]) ? 1 : 0;
        num_of_ones_x += result.X[i];
        result.Y[i] = (first.Y[i] == second.Y[i]) ? 1 : 0;
        num_of_ones_y += result.Y[i];
    }

    return result;
}

ESL_num adder_arr(ESL_num* input, int num_of_elements){
    if(num_of_elements == 0){
        printf("ERROR. Calling adder with zero inputs\n");
        exit(-1);
    }

    if(num_of_elements == 1){
        return input[0];
    }

    ESL_num result;
    int num_of_ones;
    char* nominator_terms;

    nominator_terms = (char*)malloc(num_of_elements*sizeof(char));
    for (int i = 0; i < ESL_LEN; ++i) {
        for (int j = 0; j < num_of_elements; ++j) {
            num_of_ones = 0;
            for (int k = 0; k < num_of_elements; ++k) {
                num_of_ones += (j == k) ? input[k].X[i] : input[k].Y[i];
            }
            nominator_terms[j] = (num_of_ones % 2) == 0 ? 1 : 0;
        }
        result.X[i] = nominator_terms[rand() % num_of_elements];

        num_of_ones = 0;
        for (int j = 0; j < num_of_elements; ++j) {
            num_of_ones += input[j].Y[i];
        }
        result.Y[i] = ((rand() % num_of_elements) == 0) ? ((num_of_ones % 2) == 0 ? 1 : 0) : (rand() % 2);
    }

    return result;
}

ESL_num adder_arr_1(ESL_num* input, int num_of_elements){
    if(num_of_elements == 0){
        printf("ERROR. Calling adder with zero inputs\n");
        exit(-1);
    }

    if(num_of_elements == 1){
        return input[0];
    }else if(num_of_elements == 2){
        return adder(input[0], input[1]);
    }

    int num_of_elements_log = (int)pow(2, (int)log2(num_of_elements));
    ESL_num result1 = adder(adder_arr_1(input, num_of_elements_log/2), adder_arr_1(&(input[num_of_elements_log/2]), num_of_elements_log/2));
    if(num_of_elements == num_of_elements_log){
        return result1;
    } else {
        ESL_num result2 = adder_arr_1(&(input[num_of_elements_log]), num_of_elements - num_of_elements_log);
        return adder(result1, result2);
    }
}

ESL_num adder_arr_2(ESL_num* input, int num_of_elements){
    if(num_of_elements == 0){
        printf("ERROR. Calling adder with zero inputs\n");
        exit(-1);
    }

    if(num_of_elements == 1){
        return input[0];
    }else if(num_of_elements == 2){
        return adder_2(input[0], input[1]);
    }

    int num_of_elements_log = (int)pow(2, (int)log2(num_of_elements));
    ESL_num result1 = adder_2(adder_arr_1(input, num_of_elements_log/2), adder_arr_1(&(input[num_of_elements_log/2]), num_of_elements_log/2));
    if(num_of_elements == num_of_elements_log){
        return result1;
    } else {
        ESL_num result2 = adder_arr_1(&(input[num_of_elements_log]), num_of_elements - num_of_elements_log);
        return adder_2(result1, result2);
    }
}

ESL_num adder(ESL_num first, ESL_num second) {
    //return SNG(ESL_to_double(first)+ESL_to_double(second));

    ESL_num result, half;
    int x_num = 0;
    int y_num = 0;
    for (int i = 0; i < ESL_LEN; ++i) {
        result.X[i] = ((rand() % 2) == 0) ? ((first.X[i] == second.Y[i]) ? 1 : 0) : ((first.Y[i] == second.X[i]) ? 1 : 0);
        result.Y[i] = ((rand() % 2) == 0) ? ((first.Y[i] == second.Y[i]) ? 1 : 0) : (rand() % 2);
    }
    return result;
}

ESL_num adder_2(ESL_num first, ESL_num second) {
    ESL_num result;
    int x_num = 0;
    int y_num = 0;
    int half;
    for (int i = 0; i < ESL_LEN; ++i) {
        result.X[i] = ((rand() % 2) == 0) ? ((first.X[i] == second.Y[i]) ? 1 : 0) : ((first.Y[i] == second.X[i]) ? 1 : 0);
        half = ((rand() % (int)(pow(2, BIN_LEN))) < (int)(pow(2, BIN_LEN - 1) + pow(2, BIN_LEN - 2))) ? 1 : 0;
        result.Y[i] = (((first.Y[i] + second.Y[i]) % 2) == 0) ? 1 : 0;
        result.Y[i] = (((result.Y[i] + half) % 2) == 0) ? 1 : 0;
    }
    return result;
}

ESL_num mac(ESL_num input, ESL_num weight, ESL_num initial){
    return adder_2(multiplier(input, weight), initial);
}

void read_cnn_weights(ESL_num**** weights, ESL_num* biasses, int input_channel, int output_channel, int kernel_size, char* weight_file, char* bias_file){
    double temp;

    FILE *fp_weight = fopen(weight_file, "rb");
    FILE *fp_bias = fopen(bias_file, "rb");
    if (fp_weight == NULL || fp_bias == NULL){
        printf("ERROR. %s or %s doesn't exist\n", weight_file, bias_file);
        exit(-1);
    }
    for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
        fscanf(fp_bias, "%lf\n", &temp);
        biasses[o_ch_itr] = SNG(temp);
        for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
            for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                    fscanf(fp_weight, "%lf", &temp);
                    weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr] = SNG(temp);
                }
            }
        }
    }
	fclose(fp_weight);
	fclose(fp_bias);
}


double bipolar_stochastic_divider(ESL_num input){
    int q_num_of_ones = 0;
    int max_num = pow(2, BIN_LEN);
    int approximate_num = pow(2, BIN_LEN - 1);
    int sc_approximate_num;
    int approximate_p;
    int com_approximate;
    int max_approximate;
    int min_approximate;
    for (int i = 0; i < ESL_LEN; ++i) {
        q_num_of_ones += input.Y[i];
    }
    if(q_num_of_ones < pow(2, BIN_LEN - 1)){
        for (int i = 0; i < ESL_LEN; ++i) {
            input.X[i] = 1 - input.X[i];
        }
    }
    for (int i = 0; i < ESL_LEN; ++i) {
        sc_approximate_num = ((rand() % max_num) < approximate_num) ? 1 : 0;
        approximate_p = (sc_approximate_num == input.Y[i]) ? 1 : 0;
        com_approximate = (approximate_p != input.X[i]) ? 1 : 0;
        max_approximate = (approximate_num == (max_num - 1)) ? 1 : 0;
        min_approximate = (approximate_num == 0) ? 1 : 0;
        if(com_approximate == 1){
            if(input.X[i] == 1 && max_approximate == 0){
                approximate_num += 1;
            } else if(input.X[i] == 0 && min_approximate == 0){
                approximate_num -= 1;
            }
        }
    }
    return ((double)(approximate_num) / pow(2, BIN_LEN - 1)) - 1.00000;
}

void fc_layer(ESL_num** weights, ESL_num* biasses, ESL_num* inputs, ESL_num* outputs, int input_num, int output_num, int af_num){
    int num_of_elements = 0;
    ESL_num* elements = (ESL_num*)malloc((output_num+1)*sizeof(ESL_num));

    for (int i = 0; i < output_num; i++) {
		elements[0] = biasses[i];
        num_of_elements = 1;
		for (int j = 0; j < input_num; j++) {
            elements[num_of_elements] =  multiplier(inputs[j], weights[i][j]);
            num_of_elements++;
		}

        outputs[i] = adder_arr(elements, num_of_elements);

        switch (af_num) {
            case RELU_AF:
                outputs[i] = relu_af(outputs[i]);
                break;
            case TANH_AF:
                outputs[i] = tanh_af(outputs[i]);
                break;
        }
	}
}

void read_fc_weights(ESL_num** weights, ESL_num* biasses, int input_num, int output_num, char* weight_file, char* bias_file){
    double temp;
    FILE *fp_weight = fopen(weight_file, "rb");
    FILE *fp_bias = fopen(bias_file, "rb");
    if (fp_weight == NULL || fp_bias == NULL){
        printf("ERROR. %s or %s doesn't exist\n", weight_file, bias_file);
        exit(-1);
    }
    for (int i = 0; i < output_num; i++) {
        fscanf(fp_bias, "%lf\n", &temp);
        biasses[i] = SNG(temp);
        for (int j = 0; j < input_num; j++) {
            fscanf(fp_weight, "%lf", &temp);
            weights[i][j] = SNG(temp);
        }
    }
	fclose(fp_weight);
	fclose(fp_bias);
}

int fc_soft_max(ESL_num inputs[MAX_FEATURE_NUM], int feature_num){
    int max = 0;
    double features[MAX_FEATURE_NUM];
    for (int i = 0; i < feature_num; i++) {
        features[i] = ESL_to_double(inputs[i]);
    }
    for (int i = 1; i < feature_num; i++) {
        if(features[max] < features[i])
            max = i;
    }
    return max;
}

void cnn_layer(ESL_num**** weights, ESL_num* biasses, ESL_num*** inputs, ESL_num*** outputs, int input_channel, int output_channel, int input_size, int kernel_size, int stride, int zero_pad, int af_num){
    int output_size = ((input_size + (2 * zero_pad) - kernel_size) / stride) + 1;

    int num_of_elements = 0;
    ESL_num* elements = (ESL_num*)malloc((input_channel*kernel_size*kernel_size+1)*sizeof(ESL_num));

    for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
        for (int o_r_itr = 0; o_r_itr < output_size; o_r_itr++) {
            for (int o_c_itr = 0; o_c_itr < output_size; o_c_itr++) {
                elements[0] = biasses[o_ch_itr];
                num_of_elements = 1;
        		for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
                    for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                        for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                            if((((stride*o_r_itr)+k_r_itr-zero_pad) >= 0) && (((stride*o_c_itr)+k_c_itr-zero_pad) >= 0) && (((stride*o_r_itr)+k_r_itr-zero_pad) < input_size) && (((stride*o_c_itr)+k_c_itr-zero_pad) < input_size)){
                                elements[num_of_elements] =  multiplier(
                                                                    inputs[i_ch_itr][(stride*o_r_itr)+k_r_itr-zero_pad][(stride*o_c_itr)+k_c_itr-zero_pad],
                                                                    weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]
                                                                );
                                // printf("(i[%d][%d][%d]=%lf)*(w[%d][%d][%d][%d]=%lf)=%lf(%lf)\n",
                                //     i_ch_itr, (stride*o_r_itr)+k_r_itr-zero_pad, (stride*o_c_itr)+k_c_itr-zero_pad,
                                //     ESL_to_double(inputs[i_ch_itr][(stride*o_r_itr)+k_r_itr-zero_pad][(stride*o_c_itr)+k_c_itr-zero_pad]),
                                //     o_ch_itr, i_ch_itr, k_r_itr, k_c_itr,
                                //     ESL_to_double(weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]),
                                //     ESL_to_double(inputs[i_ch_itr][(stride*o_r_itr)+k_r_itr-zero_pad][(stride*o_c_itr)+k_c_itr-zero_pad]) * ESL_to_double(weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]),
                                //     ESL_to_double(elements[num_of_elements])
                                // );
                                num_of_elements++;
                            }
                		}
                    }
        		}
                outputs[o_ch_itr][o_r_itr][o_c_itr] = adder_arr(elements, num_of_elements);
                switch (af_num) {
                    case RELU_AF:
                        outputs[o_ch_itr][o_r_itr][o_c_itr] = relu_af(outputs[o_ch_itr][o_r_itr][o_c_itr]);
                        break;
                    case TANH_AF:
                        outputs[o_ch_itr][o_r_itr][o_c_itr] = tanh_af(outputs[o_ch_itr][o_r_itr][o_c_itr]);
                        break;
                }
    		}
		}
	}

}

int A_l_B(ESL_num first, ESL_num second){
    return (ESL_to_double(first) < ESL_to_double(second)) ? 1 : 0;
}

void cnn_pool(ESL_num*** inputs, ESL_num*** outputs, int feature_channel, int input_size, int kernel_size, int stride, int zero_pad, int pool_num){
    int output_size = ((input_size + (2 * zero_pad) - kernel_size) / stride) + 1;
    ESL_num new_candidate;
    for (int ch_itr = 0; ch_itr < feature_channel; ch_itr++) {
        for (int o_r_itr = 0; o_r_itr < output_size; o_r_itr++) {
            for (int o_c_itr = 0; o_c_itr < output_size; o_c_itr++) {
                switch (pool_num) {
                    case MAX_POOL:
                        outputs[ch_itr][o_r_itr][o_c_itr] = inputs[ch_itr][stride*o_r_itr][stride*o_c_itr];
                        break;
                    case MEAN_POOL:
                        outputs[ch_itr][o_r_itr][o_c_itr] = SNG(0.000000);
                        break;
                }
                for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                    for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                        new_candidate = inputs[ch_itr][(stride*o_r_itr)+k_r_itr][(stride*o_c_itr)+k_c_itr];
                        switch (pool_num) {
                            case MAX_POOL:
                                outputs[ch_itr][o_r_itr][o_c_itr] = A_l_B(outputs[ch_itr][o_r_itr][o_c_itr], new_candidate) == 1 ? new_candidate : outputs[ch_itr][o_r_itr][o_c_itr];
                                break;
                            case MEAN_POOL:
                                outputs[ch_itr][o_r_itr][o_c_itr] = adder(new_candidate, outputs[ch_itr][o_r_itr][o_c_itr]);
                                break;
                        }
                    }
        		}
                // switch (pool_num) {
                //     case MEAN_POOL:
                //         outputs[ch_itr][o_r_itr][o_c_itr] = quantization(outputs[ch_itr][o_r_itr][o_c_itr] / (double)(kernel_size * kernel_size));
                //         break;
                // }
    		}
		}
	}
}

void print_fc_weights(ESL_num** weights, int input_num, int output_num){
    for (int i = 0; i < output_num; i++) {
        printf("|\t");
        for (int j = 0; j < input_num; j++) {
            printf("%lf\t", ESL_to_double(weights[i][j]));
            if(j != input_num - 1){
                printf(" ");
            }
        }
        printf("|\n");
    }
    printf("\n");
}

void print_fc_features(ESL_num* feature, int feature_num){
    for (int i = 0; i < feature_num; i++) {
        printf("|\t%lf\t|\n", ESL_to_double(feature[i]));
    }
    printf("\n");
}

void print_cnn_weights(ESL_num**** weights, int output_channel, int input_channel, int kernel_size){
    for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
        for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
            for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
                printf("|\t");
                for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                    printf("%lf\t", ESL_to_double(weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]));
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

void print_cnn_features(ESL_num*** feature, int feature_channel, int feature_size){
    for (int ch_itr = 0; ch_itr < feature_channel; ch_itr++) {
        for (int r_itr = 0; r_itr < feature_size; r_itr++) {
            printf("|\t");
            for (int c_itr = 0; c_itr < feature_size; c_itr++) {
                printf("%lf\t", ESL_to_double(feature[ch_itr][r_itr][c_itr]));
            }
            printf("|\n");
        }
        printf("\n");
    }
    printf("\n");
}


void print_fc_weights_file(ESL_num** weights, int input_num, int output_num, char* address){
    FILE *fp;
    fp = fopen(address, "w+");
    for (int i = 0; i < output_num; i++) {
        for (int j = 0; j < input_num; j++) {
            fprintf(fp, "%lf\n", ESL_to_double(weights[i][j]));
        }
    }
    fclose(fp);
}

void print_fc_features_file(ESL_num* feature, int feature_num, char* address){
    FILE *fp;
    fp = fopen(address, "w+");
    for (int i = 0; i < feature_num; i++) {
        fprintf(fp, "%lf\n", ESL_to_double(feature[i]));
    }
    fclose(fp);
}

void print_cnn_weights_file(ESL_num**** weights, int output_channel, int input_channel, int kernel_size, char* address){
    FILE *fp;
    fp = fopen(address, "w+");
    for (int o_ch_itr = 0; o_ch_itr < output_channel; o_ch_itr++) {
        for (int i_ch_itr = 0; i_ch_itr < input_channel; i_ch_itr++) {
            for (int k_r_itr = 0; k_r_itr < kernel_size; k_r_itr++) {
                for (int k_c_itr = 0; k_c_itr < kernel_size; k_c_itr++) {
                    fprintf(fp, "%lf\n", ESL_to_double(weights[o_ch_itr][i_ch_itr][k_r_itr][k_c_itr]));
                }
            }
        }
    }
    fclose(fp);
}

void print_cnn_features_file(ESL_num*** feature, int feature_channel, int feature_size, char* address){
    FILE *fp;
    fp = fopen(address, "w+");
    for (int ch_itr = 0; ch_itr < feature_channel; ch_itr++) {
        for (int r_itr = 0; r_itr < feature_size; r_itr++) {
            for (int c_itr = 0; c_itr < feature_size; c_itr++) {
                fprintf(fp, "%lf\n", ESL_to_double(feature[ch_itr][r_itr][c_itr]));
            }
        }
    }
    fclose(fp);
}


void cnn_to_fc(ESL_num*** cnn_feature, int cnn_feature_channel, int cnn_feature_size, ESL_num* fc_feature){
    for (int ch_itr = 0; ch_itr < cnn_feature_channel; ch_itr++) {
        for (int r_itr = 0; r_itr < cnn_feature_size; r_itr++) {
            for (int c_itr = 0; c_itr < cnn_feature_size; c_itr++) {
                fc_feature[(ch_itr*cnn_feature_size*cnn_feature_size)+(r_itr*cnn_feature_size)+c_itr] = cnn_feature[ch_itr][r_itr][c_itr];
            }
        }
    }
}

void LeNet(){

    ESL_num** fc_weights;
    fc_weights = (ESL_num**)malloc(MAX_FEATURE_NUM*sizeof(ESL_num*));
    for (int i = 0; i < MAX_FEATURE_NUM; i++) {
        fc_weights[i] = (ESL_num*)malloc(MAX_FEATURE_NUM*sizeof(ESL_num));
    }

    ESL_num* fc_biasses;
    fc_biasses = (ESL_num*)malloc(MAX_FEATURE_NUM*sizeof(ESL_num));

    ESL_num* fc_inputs;
    fc_inputs = (ESL_num*)malloc(MAX_FEATURE_NUM*sizeof(ESL_num));

    ESL_num* fc_outputs;
    fc_outputs = (ESL_num*)malloc(MAX_FEATURE_NUM*sizeof(ESL_num));

    int fc_input_num;
    int fc_output_num;

    unsigned char input_labels[MAX_TEST_SAMPLES];

    ESL_num**** cnn_weights;
    cnn_weights = (ESL_num****)malloc(MAX_CHANNEL*sizeof(ESL_num***));
    for (int i = 0; i < MAX_CHANNEL; i++) {
        cnn_weights[i] = (ESL_num***)malloc(MAX_CHANNEL*sizeof(ESL_num**));
        for (int j = 0; j < MAX_CHANNEL; j++) {
            cnn_weights[i][j] = (ESL_num**)malloc(MAX_KERNEL_SIZE*sizeof(ESL_num*));
            for (int k = 0; k < MAX_KERNEL_SIZE; k++) {
                cnn_weights[i][j][k] = (ESL_num*)malloc(MAX_KERNEL_SIZE*sizeof(ESL_num));
            }
        }
    }

    ESL_num* cnn_biasses;
    cnn_biasses = (ESL_num*)malloc(MAX_CHANNEL*sizeof(ESL_num));

    ESL_num*** cnn_inputs;
    cnn_inputs = (ESL_num***)malloc(MAX_CHANNEL*sizeof(ESL_num**));
    for (int i = 0; i < MAX_CHANNEL; i++) {
        cnn_inputs[i] = (ESL_num**)malloc(MAX_FEATURE_SIZE*sizeof(ESL_num*));
        for (int j = 0; j < MAX_FEATURE_SIZE; j++) {
            cnn_inputs[i][j] = (ESL_num*)malloc(MAX_FEATURE_SIZE*sizeof(ESL_num));
        }
    }

    ESL_num*** cnn_outputs;
    cnn_outputs = (ESL_num***)malloc(MAX_CHANNEL*sizeof(ESL_num**));
    for (int i = 0; i < MAX_CHANNEL; i++) {
        cnn_outputs[i] = (ESL_num**)malloc(MAX_FEATURE_SIZE*sizeof(ESL_num*));
        for (int j = 0; j < MAX_FEATURE_SIZE; j++) {
            cnn_outputs[i][j] = (ESL_num*)malloc(MAX_FEATURE_SIZE*sizeof(ESL_num));
        }
    }

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
    num_of_tests = 1;

    reset_cal();

    for (int itr = 0; itr < num_of_tests; itr++) {

        cnn_output_channel = 6;
        cnn_input_channel = 1;
        cnn_input_size = 28;
        cnn_output_size = 28;
        cnn_kernel_size = 5;
        cnn_stride = 1;
        cnn_zero_padd = 2;
        cnn_af_type = RELU_AF;
        read_cnn_inputs(cnn_inputs, input_labels, cnn_input_channel, cnn_input_size, num_of_tests, IMAGE_ADDRESS, LABEL_ADDRESS);
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

#if(DEBUG_DEF_FILE == 1)
        print_cnn_weights_file(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size, "./Network_outputs/L12/CNN1_Weights");
        print_cnn_features_file(cnn_inputs, cnn_input_channel, cnn_input_size, "./Network_outputs/L12/CNN1_Inputs");
        print_cnn_features_file(cnn_outputs, cnn_output_channel, cnn_output_size, "./Network_outputs/L12/CNN1_Outputs");
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

#if(DEBUG_DEF_FILE == 1)
        print_cnn_features_file(cnn_inputs, cnn_output_channel, cnn_output_size, "./Network_outputs/L12/CNN1_Final_Outputs");
#endif


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

#if(DEBUG_DEF_FILE == 1)
        print_cnn_weights_file(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size, "./Network_outputs/L12/CNN2_Weights");
        print_cnn_features_file(cnn_outputs, cnn_output_channel, cnn_output_size, "./Network_outputs/L12/CNN2_Outputs");
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

#if(DEBUG_DEF_FILE == 1)
        print_cnn_features_file(cnn_inputs, cnn_output_channel, cnn_output_size, "./Network_outputs/L12/CNN2_Final_Outputs");
#endif

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
        printf("Layer 3: CNN:\n");
        printf("CNN weights:\n");
        print_cnn_weights(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size);
        printf("CNN Inputs:\n");
        print_cnn_features(cnn_inputs, cnn_input_channel, cnn_input_size);
        printf("CNN Outputs:\n");
        print_cnn_features(cnn_outputs, cnn_output_channel, cnn_output_size);
#endif

#if(DEBUG_DEF_FILE == 1)
        print_cnn_weights_file(cnn_weights, cnn_output_channel, cnn_input_channel, cnn_kernel_size, "./Network_outputs/L12/CNN3_Weights");
        print_cnn_features_file(cnn_outputs, cnn_output_channel, cnn_output_size, "./Network_outputs/L12/CNN3_Outputs");
#endif


        cnn_to_fc(cnn_outputs, cnn_output_channel, cnn_output_size, fc_inputs);



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

#if(DEBUG_DEF_FILE == 1)
        print_fc_weights_file(fc_weights, fc_input_num, fc_output_num, "./Network_outputs/L12/FC1_Weights");
        print_fc_features_file(fc_outputs, fc_output_num, "./Network_outputs/L12/FC1_Outputs");
#endif

        result_index = fc_soft_max(fc_outputs, fc_output_num);
        expected_index = input_labels[itr];
// #if(DEBUG_DEF == 1)
        printf("itr: %d, expected: %d, result: %d\n", itr, expected_index, result_index);
// #endif

#if(RESULT_SHOW == 1)
        if(itr % 10 == 0){
            printf("\r%d%% completed!", itr/10);
            fflush(stdout);
        }
#endif

        if(result_index == expected_index){
            accuracy += 1;
        }

    }

#if(RESULT_SHOW == 1)
    printf("accuracy: %d/%d\n", accuracy, num_of_tests);
#endif
    reset_cal();

    free(fc_weights);
    free(fc_biasses);
    free(fc_inputs);
    free(fc_outputs);
    free(cnn_weights);
    free(cnn_biasses);
    free(cnn_inputs);
    free(cnn_outputs);

}

void mult_accuracy(){

    int max_num = pow(2, BIN_LEN);

    double num_double_1;
    double num_double_2;
    ESL_num num_ESL_1;
    ESL_num num_ESL_2;
    double golden_answer;
    double answer;
    double diff = 0;
    double curr_diff = 0;

    diff = 0;
    for (int i = 0; i < 1000; ++i){
        num_double_1 = ((double)(rand() & (max_num - 1)) / pow(2, FRACTION_LEN)) - pow(2, INTEGER_LEN);
        num_double_2 = ((double)(rand() & (max_num - 1)) / pow(2, FRACTION_LEN)) - pow(2, INTEGER_LEN);
        num_ESL_1 = SNG(num_double_1);
        num_ESL_2 = SNG(num_double_2);
        answer = ESL_to_double(multiplier(num_ESL_1, num_ESL_2));
        golden_answer = ESL_to_double(num_ESL_1) * ESL_to_double(num_ESL_2);
        curr_diff = (golden_answer - answer) * (golden_answer - answer);
        diff += curr_diff;
    }

    printf("%lf\n", sqrt(diff/999.00000));

}

void one_sng_accuracy(){
    double num_double[1000];
    ESL_num answer;

    double diff = 0;
    double curr_diff = 0;

    diff = 0;
    for (int i = 0; i < 1000; ++i){
        answer = SNG(1.000000000);
        curr_diff = absolute(1.0000000 - ESL_to_double(answer));
        diff += curr_diff;
    }
    printf("%lf\n", diff/1000.00000);

}

void sng_accuracy(){

    double diff = 0;
    double curr_diff = 0;

    ESL_num answer;

    // for (double i = -3.99; i < 4.00; i += 0.01) {
    //     printf("%lf\n", i);
    // }

    // printf("allahoakbar\n\n\n");

    for (double i = -3.99; i < 4.00; i += 0.01) {
        diff = 0;
        for (int j = 0; j < 1000; ++j){
            answer = SNG(i);
            curr_diff = (i - ESL_to_double(answer)) * (i - ESL_to_double(answer));
            if(curr_diff < 0)
                exit(-2);
            diff += curr_diff;
        }
        printf("%lf\n", sqrt(diff/999.00000));
    }


}

void one_mult_accuracy(){
    ESL_num num_ESL;
    ESL_num answer;

    double diff = 0;
    double curr_diff = 0;

    for (int i = 0; i < 1000; ++i){
        answer = multiplier(SNG(1.00000), SNG(1.00000));
        curr_diff = absolute(1.000000 - (4.00000 * ESL_to_double(answer)));
        diff += curr_diff;
    }
    printf("%lf\n", diff/1000.00000);

}

void bipolar_divider_accuracy(){
    double num_double[1000];
    double answer, golden_answer;

    double diff = 0;
    double curr_diff = 0;

    ESL_num converted;

    for (double j = 0.99; j > -1; j = j - 0.01){
        printf("%lf\n", j);
    }
    // return 0;
    printf("\n\n\n\n\ndiifs:\n\n\n\n");

    for (double j = 0.99; j > -1; j = j - 0.01){

        diff = 0;
        for (int i = 0; i < 1000; ++i){
            converted = SNG(j);
            golden_answer = ESL_to_double(converted);
            if(golden_answer == 0){
                i--;
                continue;
            }
            answer = bipolar_stochastic_divider(converted);
            curr_diff = absolute((golden_answer - answer) / golden_answer);
            diff += curr_diff;
            //printf("g[%lf], a[%lf]\n", golden_answer, answer);
        }
        printf("%lf\n", diff/1000.000000000);
    }
}

double binary_random_generator(){
    int max_num = pow(2, BIN_LEN);
    return ((double)(rand() & (max_num - 1)) - pow(2, BIN_LEN - 1)) / pow(2, FRACTION_LEN);
}

void add_accuracy(){
    double num_double_1[1000];
    double num_double_2[1000];
    ESL_num first_num;
    ESL_num second_num;
    ESL_num result1;
    ESL_num result2;

    double diff_1 = 0;
    double diff_2 = 0;
    double curr_diff_1 = 0;
    double curr_diff_2 = 0;

    for (double j = 1.8; j > 0; j = j - 0.01){
        printf("%lf\n", j);
    }
    printf("\n\n\n\n\ndiifs:\n\n\n\n");

    for (int i = 0; i < 1000; ++i){
        num_double_1[i] = binary_random_generator();
        num_double_2[i] = binary_random_generator();
        if(absolute(num_double_1[i] + num_double_2[i]) >= 4 || absolute(num_double_1[i] + num_double_2[i]) == 0){
            i--;
            continue;
        }
    }

    diff_1 = 0;
    diff_2 = 0;
    for (int i = 0; i < 1000; ++i){
        first_num = SNG(num_double_1[i]);
        second_num = SNG(num_double_2[i]);
        result1 = adder(first_num, second_num);
        result2 = adder_2(first_num, second_num);
        curr_diff_1 = absolute((absolute(num_double_1[i] + num_double_2[i]) - absolute(ESL_to_double(result1))) / (num_double_1[i] + num_double_2[i]));
        curr_diff_2 = absolute((absolute(num_double_1[i] + num_double_2[i]) - absolute(ESL_to_double(result2))) / (num_double_1[i] + num_double_2[i]));
        diff_1 += curr_diff_1;
        diff_2 += curr_diff_2;
        printf("%lf/%lf + %lf/%lf = %lf/%lf (%lf) result1: %lf/%lf (%lf) (%lf) result2: %lf/%lf (%lf) (%lf)\n",
            ESL_to_double_nominator(first_num, 1), ESL_to_double_nominator(first_num, 0),
            ESL_to_double_nominator(second_num, 1), ESL_to_double_nominator(second_num, 0),
            ESL_to_double_nominator(first_num, 1) * ESL_to_double_nominator(second_num, 0) + ESL_to_double_nominator(first_num, 0) * ESL_to_double_nominator(second_num, 1),
            ESL_to_double_nominator(first_num, 0) * ESL_to_double_nominator(second_num, 0),
            ESL_to_double(first_num) + ESL_to_double(second_num),
            ESL_to_double_nominator(result1, 1), ESL_to_double_nominator(result1, 0),
            curr_diff_1,
            ESL_to_double(result1),
            ESL_to_double_nominator(result2, 1), ESL_to_double_nominator(result2, 0),
            ESL_to_double(result2),
            curr_diff_2
        );

    }
    printf("%lf\n", diff_1/1000.00000);
    printf("%lf\n", diff_2/1000.00000);

}

void add_arr_accuracy(){
    double num_double[40];
    ESL_num num_ESL[40];
    double golden_result;

    ESL_num result;
    double diff = 0;
    double curr_diff = 0;

    for (int num_of_elements = 2; num_of_elements <= 40; ++num_of_elements) {
        diff = 0;
        for (int test_number = 0; test_number < 1000; ++test_number){
            golden_result = 0;
            for (int i = 0; i < num_of_elements; ++i){
                num_double[i] = binary_random_generator() / (double)(num_of_elements);
                num_ESL[i] = SNG(num_double[i]);
                golden_result += ESL_to_double(num_ESL[i]);
            }
            if(golden_result == 0){
                test_number--;
                continue;
            }
            result = adder_arr(num_ESL, num_of_elements);

            curr_diff = (golden_result - ESL_to_double(result)) * (golden_result - ESL_to_double(result));
            diff += curr_diff;
        }
        printf("#of elements: %d, Error1: %lf\n", num_of_elements, sqrt(diff/999.00000));
    }

    for (int num_of_elements = 2; num_of_elements <= 30; ++num_of_elements) {
        diff = 0;
        for (int test_number = 0; test_number < 1000; ++test_number){
            golden_result = 0;
            for (int i = 0; i < num_of_elements; ++i){
                num_double[i] = binary_random_generator() / (double)(num_of_elements);
                num_ESL[i] = SNG(num_double[i]);
                golden_result += ESL_to_double(num_ESL[i]);
            }
            if(golden_result == 0){
                test_number--;
                continue;
            }
            result = adder_arr_1(num_ESL, num_of_elements);

            curr_diff = (golden_result - ESL_to_double(result)) * (golden_result - ESL_to_double(result));
            diff += curr_diff;
        }
        printf("#of elements: %d, Error2: %lf\n", num_of_elements, sqrt(diff/999.00000));
    }

    for (int num_of_elements = 2; num_of_elements <= 30; ++num_of_elements) {
        diff = 0;
        for (int test_number = 0; test_number < 1000; ++test_number){
            golden_result = 0;
            for (int i = 0; i < num_of_elements; ++i){
                num_double[i] = binary_random_generator() / (double)(num_of_elements);
                num_ESL[i] = SNG(num_double[i]);
                golden_result += ESL_to_double(num_ESL[i]);
            }
            if(golden_result == 0){
                test_number--;
                continue;
            }
            result = adder_arr_2(num_ESL, num_of_elements);

            curr_diff = (golden_result - ESL_to_double(result)) * (golden_result - ESL_to_double(result));
            diff += curr_diff;
        }
        printf("#of elements: %d, Error3: %lf\n", num_of_elements, sqrt(diff/999.00000));
    }

}

int main(){
    LeNet();
    //add_arr_accuracy();
    //bipolar_divider_accuracy();
    //add_accuracy();
    //mult_accuracy();
    //multiplier(SNG(1), SNG(1))
    //for (int i = 0; i < 10; i++) {
    //    printf("answer: %lf\n", ESL_to_double(multiplier(SNG(-2.656250), SNG(1.156250 ))));
    //}
    //sng_accuracy();
    //return 0;

    /*
    ESL_num result1, result2;
    ESL_num first_num, second_num;
    for (int i = 0; i < 10; i++) {
        first_num = SNG(1);
        second_num = SNG(1);
        result1 = adder(first_num, second_num);
        result2 = adder_2(first_num, second_num);
        printf("%lf/%lf + %lf/%lf = %lf/%lf (%lf)   result1: %lf/%lf (%lf)   result2: %lf/%lf (%lf)\n",
            ESL_to_double_nominator(first_num, 1), ESL_to_double_nominator(first_num, 0),
            ESL_to_double_nominator(second_num, 1), ESL_to_double_nominator(second_num, 0),
            ESL_to_double_nominator(first_num, 1) * ESL_to_double_nominator(second_num, 0) + ESL_to_double_nominator(first_num, 0) * ESL_to_double_nominator(second_num, 1),
            ESL_to_double_nominator(first_num, 0) * ESL_to_double_nominator(second_num, 0),
            ESL_to_double(first_num) + ESL_to_double(second_num),
            ESL_to_double_nominator(result1, 1), ESL_to_double_nominator(result1, 0),
            ESL_to_double(result1),
            ESL_to_double_nominator(result2, 1), ESL_to_double_nominator(result2, 0),
            ESL_to_double(result2)
        );
    }
    */
    return 0;
}
