import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Config(object):
    def __init__(self):
        # the length of frames
        self.frame_length = 32
        # the number of joints
        self.joint_number = 22
        # the dimension of joints
        self.joint_dim = 3
        # the number of coarse class
        self.class_coarse = 14
        # the number of fine class
        self.class_fine = 28


def short_pose_difference(skeleton):
    height = skeleton.get_shape()[1]
    width = skeleton.get_shape()[2]
    short_skeleton = tf.subtract(skeleton[:, 1:, ...], skeleton[:, 0:-1, ...])
    short_skeleton = tf.image.resize(short_skeleton, size=[height, width], method='nearest', antialias=True)
    return short_skeleton


def point_process(pose, frame_length):
    pose_diff = Lambda(lambda skeleton: short_pose_difference(skeleton))(pose)
    pose_diff = Reshape((frame_length, -1))(pose_diff)
    return pose_diff


def edge_process(pose, frame_length):
    pose_edge = Reshape((frame_length, -1))(pose)
    return pose_edge


def normalize_pose(skeleton):
    height = skeleton.get_shape()[1]
    width = skeleton.get_shape()[2]
    normalize_skeleton = tf.subtract(skeleton, tf.expand_dims(skeleton[:, :, 0, ...], 2))
    normalize_skeleton = tf.image.resize(normalize_skeleton, size=[height, width], method='nearest', antialias=True)
    return normalize_skeleton


def normalization_process(pose):
    pose_normalization = Lambda(lambda skeleton: normalize_pose(skeleton))(pose)
    return pose_normalization


def convolution_block(feature, filters, kernel, dilation=1):
    feature = SeparableConv1D(filters=filters, kernel_size=kernel, padding='same', use_bias=True, dilation_rate=dilation)(feature)
    feature = BatchNormalization()(feature)
    feature = LeakyReLU(alpha=0.2)(feature)
    feature = SpatialDropout1D(rate=0.1)(feature)
    return feature


def connect_block(feature, filters):
    feature = Dense(filters, use_bias=False)(feature)
    feature = LeakyReLU(alpha=0.2)(feature)
    feature = Dropout(rate=0.5)(feature)
    return feature


def msmh_attention_module(mix_feature, filters, kernel, num_heads=6, key_dim=8):
    large_feature = SeparableConv1D(filters=filters, kernel_size=kernel, padding='same', use_bias=True, dilation_rate=3)(mix_feature)
    # generate large feature
    level_large = BatchNormalization()(large_feature)
    level_large = LeakyReLU(alpha=0.1)(level_large)
    # add medium feature
    medium_feature = Add()([large_feature, mix_feature])
    medium_feature = SeparableConv1D(filters=filters, kernel_size=kernel, padding='same', use_bias=True, dilation_rate=2)(medium_feature)
    # generate medium feature
    level_medium = BatchNormalization()(medium_feature)
    level_medium = LeakyReLU(alpha=0.1)(level_medium)
    level_medium = MaxPooling1D(pool_size=2)(level_medium)
    # add small feature
    small_feature = Add()([medium_feature, mix_feature])
    small_feature = SeparableConv1D(filters=filters, kernel_size=kernel, padding='same', use_bias=True, dilation_rate=1)(small_feature)
    # generate small feature
    level_small = BatchNormalization()(small_feature)
    level_small = LeakyReLU(alpha=0.1)(level_small)
    level_small = MaxPooling1D(pool_size=4)(level_small)
    # multi-scale multi-head attention
    key = Concatenate(axis=1)([level_large, level_medium, level_small])
    weighted_feature = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(mix_feature, key)
    return weighted_feature

def hierarchical_attention_module(inter_feature, intra_feature, filters, kernel=3, num_heads=6, key_dim=8):
    # inter frame feature
    inter_ts = SeparableConv1D(filters=filters, kernel_size=kernel, padding='same', use_bias=False)(inter_feature)
    inter_ts = BatchNormalization()(inter_ts)
    inter_ts = LeakyReLU(alpha=0.1)(inter_ts)
    inter_ts = SpatialDropout1D(rate=0.2)(inter_ts)
    # intra frame feature
    intra_ts = SeparableConv1D(filters=filters, kernel_size=kernel, padding='same', use_bias=False)(intra_feature)
    intra_ts = BatchNormalization()(intra_ts)
    intra_ts = LeakyReLU(alpha=0.1)(intra_ts)
    intra_ts = SpatialDropout1D(rate=0.2)(intra_ts)
    # T*T attention
    t_attention = Dot(axes=(2, 2))([inter_ts, intra_ts])
    # S*S attention
    s_attention = Dot(axes=(1, 1))([inter_ts, intra_ts])
    # multi-scale multi-head attention
    mix_feature = Add()([inter_ts, intra_ts])
    msmh_feature = msmh_attention_module(mix_feature, filters, kernel)
    weighted_t = Dot(axes=(2, 1))([t_attention, msmh_feature])
    weighted_s = Dot(axes=(2, 1))([msmh_feature, s_attention])
    weighted_result = Add()([weighted_t, weighted_s])
    inter_result = Add()([weighted_result, inter_ts])
    intra_result = Add()([weighted_result, intra_ts])
    return weighted_result, inter_result, intra_result


def hourglass_block(feature, filters, rate, name):
    layer1_feature = Conv2D(filters, kernel_size=3, strides=1, padding='same', dilation_rate=rate, name=name)(feature)
    layer2_feature = BatchNormalization()(layer1_feature)
    layer3_feature = LeakyReLU(alpha=0.2)(layer2_feature)
    return layer3_feature


def hourglass_net(key_points):
    # Encoder Block
    key_points = GaussianNoise(stddev=0.01)(key_points)
    encoder_layer1 = hourglass_block(key_points, 16, 3, "encoder_layer1")
    encoder_layer2 = MaxPooling2D(pool_size=2, strides=2)(encoder_layer1)
    encoder_layer3 = hourglass_block(encoder_layer2, 32, 2, "encoder_layer2")
    encoder_layer4 = MaxPooling2D(pool_size=2, strides=2)(encoder_layer3)
    encoder_layer5 = hourglass_block(encoder_layer4, 64, 1, "encoder_layer3")
    encoder_layer6 = MaxPooling2D(pool_size=2, strides=2)(encoder_layer5)

    # Middle Block
    middle_layer = dimension_attention_module(encoder_layer6, encoder_layer6)

    # Decoder Block
    decoder_layer1 = UpSampling2D(size=2, interpolation='nearest')(middle_layer)
    sum_feature_1 = Add()([decoder_layer1, encoder_layer5])
    decoder_layer2 = hourglass_block(sum_feature_1, 32, 3, "decoder_layer1")

    decoder_layer3 = UpSampling2D(size=2, interpolation='nearest')(decoder_layer2)
    skip_layer1 = UpSampling2D(size=4, interpolation='nearest')(middle_layer)
    skip_layer1 = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(skip_layer1)
    sum_feature_2 = Add()([decoder_layer3, encoder_layer3, skip_layer1])
    decoder_layer4 = hourglass_block(sum_feature_2, 16, 2, "decoder_layer2")

    decoder_layer5 = UpSampling2D(size=2, interpolation='nearest')(decoder_layer4)
    skip_layer2 = UpSampling2D(size=8, interpolation='nearest')(middle_layer)
    skip_layer2 = Conv2D(16, kernel_size=1, strides=1, padding='same', activation='relu')(skip_layer2)
    sum_feature_3 = Add()([decoder_layer5, encoder_layer1, skip_layer2])
    decoder_layer6 = hourglass_block(sum_feature_3, 6, 1, "decoder_layer3")

    decoded_result = Conv2D(3, kernel_size=1, strides=1, activation='sigmoid', name='reconstruction',
                            padding='same')(decoder_layer6)
    
    return decoded_result, encoder_layer1, encoder_layer3, encoder_layer5

def dimension_attention_module(attention_feature, target_feature):
    feature_t = Reshape((attention_feature.shape[1], attention_feature.shape[2]*attention_feature.shape[3]))(attention_feature)
    feature_t = Dense(attention_feature.shape[1], activation='gelu')(feature_t)
    feature_s = Reshape((attention_feature.shape[2], attention_feature.shape[1]*attention_feature.shape[3]))(attention_feature)
    feature_s = Dense(attention_feature.shape[2], activation='gelu')(feature_s)
    feature_c = Reshape((attention_feature.shape[3], attention_feature.shape[1]*attention_feature.shape[2]))(attention_feature)
    feature_c = Dense(attention_feature.shape[3], activation='gelu')(feature_c)
    weighted_t = Dot(axes=(1, 1))([feature_t, target_feature])
    weighted_s = Permute((1, 3, 2))(target_feature)
    weighted_s = Dot(axes=(3, 1))([weighted_s, feature_s])
    weighted_s = Permute((1, 3, 2))(weighted_s)
    weighted_c = Dot(axes=(3, 1))([target_feature, feature_c])
    result_feature = Add()([weighted_t, weighted_s, weighted_c])
    return result_feature


def stair_block(original_feature, low_feature, middle_feature, high_feature, out_channels=16):
    # Low Layer Operation
    low_layer1 = concatenate([original_feature, low_feature])
    low_layer2 = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same', dilation_rate=1)(low_layer1)
    low_layer3 = BatchNormalization()(low_layer2)
    low_layer4 = ReLU()(low_layer3)
    low_layer5 = SeparableConv2D(out_channels*2, kernel_size=3, strides=1, padding='same', dilation_rate=1)(low_layer4)
    low_layer6 = BatchNormalization()(low_layer5)
    low_layer7 = ReLU()(low_layer6)
    low_to_middle = MaxPooling2D(pool_size=2, strides=2)(low_layer7)

    # Middle Layer Operation
    middle_layer1 = MaxPooling2D(pool_size=2, strides=2)(low_layer4)
    middle_layer2 = concatenate([middle_layer1, middle_feature])
    middle_layer3 = SeparableConv2D(out_channels*2, kernel_size=3, strides=1, padding='same', dilation_rate=1)(middle_layer2)
    middle_layer4 = BatchNormalization()(middle_layer3)
    middle_layer5 = ReLU()(middle_layer4)
    middle_layer6 = concatenate([low_to_middle, middle_layer5])
    middle_layer7 = SeparableConv2D(out_channels*4, kernel_size=3, strides=1, padding='same', dilation_rate=1)(middle_layer6)
    middle_layer8 = BatchNormalization()(middle_layer7)
    middle_layer9 = ReLU()(middle_layer8)
    middle_to_high = MaxPooling2D(pool_size=2, strides=2)(middle_layer9)
    
    # High Layer Operation
    high_layer1 = MaxPooling2D(pool_size=2, strides=2)(middle_layer2)
    high_layer2 = concatenate([high_layer1, high_feature])
    high_layer3 = SeparableConv2D(out_channels*4, kernel_size=3, strides=1, padding='same', dilation_rate=1)(high_layer2)
    high_layer4 = BatchNormalization()(high_layer3)
    high_layer5 = ReLU()(high_layer4)
    high_layer6 = concatenate([middle_to_high, high_layer5])
    high_layer7 = SeparableConv2D(out_channels*8, kernel_size=3, strides=1, padding='same', dilation_rate=1)(high_layer6)
    high_layer8 = BatchNormalization()(high_layer7)
    high_layer9 = ReLU()(high_layer8)

    return high_layer9


def stair_head(feature, out_channels=3, kernel=5, dilation=1):
    residual_layer1 = SeparableConv2D(out_channels, kernel_size=kernel, strides=1, padding='same', dilation_rate=dilation)(feature)
    residual_layer2 = BatchNormalization()(residual_layer1)
    residual_layer3 = Add()([feature, residual_layer2])
    residual_layer4 = ReLU()(residual_layer3)
    residual_layer5 = Dropout(rate=0.1)(residual_layer4)
    return residual_layer5


def stair_net(original_feature, low_feature, middle_feature, high_feature, class_number):
    stair_layer1 = stair_head(original_feature)
    stair_layer2 = stair_block(stair_layer1, low_feature, middle_feature, high_feature, 16)
    classify_layer1 = GlobalMaxPooling2D()(stair_layer2)
    classify_layer2 = connect_block(classify_layer1, 96)
    classify_layer3 = Dense(class_number, activation='softmax', name='spatial')(classify_layer2)
    return classify_layer3


def build_net(frame_length=32, joint_number=22, joint_dim=3, class_number=14):
    pose_point = Input(name='pose_point', shape=(frame_length, joint_number, joint_dim))
    pose_edge = Input(name='pose_edge', shape=(frame_length, joint_number - 1, joint_dim))
    pose_norm = Input(name='pose_norm', shape=(frame_length, joint_number+2, joint_dim))

    inter_feature = point_process(pose_point, frame_length)
    intra_feature = edge_process(pose_edge, frame_length)

    _, inter_1, intra_1 = hierarchical_attention_module(inter_feature, intra_feature, 64)
    inter_1 = MaxPooling1D(pool_size=2)(inter_1)
    intra_1 = MaxPooling1D(pool_size=2)(intra_1)
    weighted_feature, _, _ = hierarchical_attention_module(inter_1, intra_1, 96)

    weighted_feature = GlobalMaxPooling1D()(weighted_feature)
    weighted_feature = connect_block(weighted_feature, 96)
    temporal_result = Dense(class_number, activation='softmax', name="temporal")(weighted_feature)

    reconstruction_result, hourglass_feature1, hourglass_feature2, hourglass_feature3 = hourglass_net(pose_norm)
    spatial_result = stair_net(pose_norm, hourglass_feature1, hourglass_feature2, hourglass_feature3, class_number)
    result = Average(name="result")([temporal_result, spatial_result])
    model = Model(inputs=[pose_point, pose_edge, pose_norm],
                  outputs=[result, temporal_result, spatial_result, reconstruction_result])
    return model


# Method 1
# reference: https://github.com/tensorflow/tensorflow/issues/32809
def get_flops(model, model_inputs) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.
        """
        # if not hasattr(model, "model"):
        #     raise wandb.Error("self.model must be set before using this method.")

        if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        batch_size = 1
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in model_inputs
        ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        tf.compat.v1.reset_default_graph()

        # convert to GFLOPs
        return (flops.total_float_ops / 1e9) / 2

# # Method 2
# # reference: https://github.com/tokusumi/keras-flops/blob/master/notebooks/flops_calculation_tfkeras.ipynb
# def get_flops(model, batch_size=None):
#     if batch_size is None:
#         batch_size = 1

#     real_model = tf.function(model).get_concrete_function((tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype),
#                                                         tf.TensorSpec([batch_size] + model.inputs[1].shape[1:], model.inputs[1].dtype),
#                                                         tf.TensorSpec([batch_size] + model.inputs[2].shape[1:], model.inputs[2].dtype)))
#     frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

#     run_meta = tf.compat.v1.RunMetadata()
#     opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#     flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
#                                             run_meta=run_meta, cmd='op', options=opts)
#     # return GFLOPs
#     return (flops.total_float_ops / 1e9) / 2


if __name__ == '__main__':
    config = Config()
    network = build_net(config.frame_length, config.joint_number, config.joint_dim, config.class_coarse)
    network.summary()
    
    ######### We provide two methods to calculate GFLOPs. ###########

    # Method 1
    pose_point = tf.constant(np.random.randn(1,32, 22, 3))
    pose_edge = tf.constant(np.random.randn(1,32, 21, 3))
    pose_norm = tf.constant(np.random.randn(1,32, 24, 3))
    gflops = get_flops(network, [pose_point, pose_edge, pose_norm])
    print(f"GFLOPs: {gflops}")
    
    # # Method 2
    # gflops = get_flops(network)
    # print(f"GFLOPs: {gflops}")







