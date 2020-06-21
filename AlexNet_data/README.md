# DeltaNN Weight Extractor for AlexNet

This repo extracts weights and indices in the _bvlc_alexnet.npy_ file according to the DeltaNN computation. [Download this file](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) into this directory.

Execution time: 20 minutes on my computer on one core. I didn't have enough time to multi-thread it, but, I think we'll have to for VGG16 as it has much more weights.

The following items are the parameters to this code:

- **WEIGHT_ABS_LEN**: The length of the input features and weights in the DeltaNN. 1 bit for sign, and 7 bits for fraction is considered in this version of code. There is not any integer weight as all of the weights and biasses are less than 1.
- **WEIGHT_DELTA_LEN**: The length of the delta weights in V1.
- **WEIGHT_DELTA_LEN_LOG**: The length of the delta weights in V2.
- **WEIGHT_NUM_LEN**: We show number of unique weight repetitions through the blocks of this parameter bits. This is for V1.
- **WEIGHT_NUM_LEN_LOG**: We show number of unique weight repetitions through the blocks of this parameter bits. This is for V2.
- **T_input_channel**: Number of input channels in a tile.
- **T_output_channel**: Number of output channels in a tile.
- **T_output_size**: Height and width of the output features covered in a tile.
- **T_input_size_cons**: Max height and width of the input features covered in a tile. The real **T_input_size** is found according to the kernel size and output size for each layer.


* DO NOT CHANGE THESE PARAMETERS. They are investigated for minimum memory storage.
* For further information around these parameters refer to the ΔNN doc.

Here are the outputs of this file:

- **bvlc_alexnet_v1.npy**: Modified version of weights and indices according to the DeltaNN V1 implementation. In this version, weights and biasses are only quantized since DeltaNN V1 does not change them.
- **bvlc_alexnet_v2.npy**: Modified version of weights and indices according to the DeltaNN V2 implementation. In addition to the above modifications, we ignore the zero weights, sort the weights in each tile and each input channel in tile, find the unique weights, find the deltas and their nearest power of 2, and store the weights that are used in a real V2 implementation. 
- **convX/fcX_index_1d_tiled.txt**: 1D indices for each tile are separated with a \n. Each input channel's 1D indices are also show in 1 line. We use 1D indices for delta index calculation.
 - For CNNs: Although it doesn't impact the A* algorithm approach, remember that 1D indices are found with this order of dimensions: [kernel row][kernel column][output channel]. This order is consider so that the indices that are related to 1 unique weight are not ordered based on their [output channel] dimension, and thus, bubbles are reduced.
 - For FCs: 1D indices are equal to the output channel. Collisions are detected using these indices in each tile.
- **convX_index_oc_tiled.txt**: Output channels corresponding to each 1D index in the _convX_index_1d_tiled_ file. Collisions are detected through these indices in a CNN layer.
- **convX_index_kr_tiled.txt**: Kernel rows corresponding to each 1D index in the _convX_index_1d_tiled_ file. Although they are not used in the collision detection, we must encode them in the absolute indices in this order: [output channel][kernel row][kernel column].
- **convX_index_kr_tiled.txt**: Kernel columns corresponding to each 1D index in the _convX_index_1d_tiled_ file. Description same as the above item.
- **ConvX_fcX_index_ic_tiled.txt**: Input channels corresponding to each 1D index in the _convX_index_1d_tiled_ file. This file is not used in A* algorithm and is only for sanity check.
- **alexnet_info.csv**: Contains memory requirement information for each layer in a _.csv_ file.
