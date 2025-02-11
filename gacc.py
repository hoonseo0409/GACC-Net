# import imputation_core.cgain.partial_conv as partial_conv
import numpy as np
import tensorflow as tf
# import cv2

# from scipy import ndimage
# from matplotlib import pyplot as plt
from tqdm import tqdm
import utilsforminds.helpers as helpers
from time import time
assert(tf.executing_eagerly())
from copy import deepcopy

from keras.layers import Dropout, BatchNormalization, Concatenate, RepeatVector, Dense, LSTM, Input, LeakyReLU, Lambda, GRU, SimpleRNN, Flatten, Conv2DTranspose, ConvLSTM2D, MaxPooling3D, TimeDistributed, Reshape, Subtract, Add, Multiply, MultiHeadAttention, LayerNormalization, Conv2D, MaxPooling2D, Convolution2D, Conv3D, Embedding, Layer
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from keras.models import Model
from keras.utils import plot_model
import keras
import keras.backend as backend
from keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras import Sequential

from random import random, randint, sample
import utilsforminds
from typing import Callable

keras_layer_dict = {"Conv2D": Conv2D, "MaxPooling2D": MaxPooling2D, "Dense": Dense, "Dropout": Dropout, "BatchNormalization": BatchNormalization, "LSTM": LSTM, "GRU": GRU, "Flatten": Flatten, "Conv2DTranspose": Conv2DTranspose, "ConvLSTM2D": ConvLSTM2D, "MaxPooling3D": MaxPooling3D, "TimeDistributed": TimeDistributed, "Reshape": Reshape, "RandomFourierFeatures": RandomFourierFeatures, "MultiHeadAttention": MultiHeadAttention, "LayerNormalization": LayerNormalization}

if False:
    class PositionEmbeddingFixedWeights(Layer):
        def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
            super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
            word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)   
            position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)                                          
            self.word_embedding_layer = Embedding(
                input_dim=vocab_size, output_dim=output_dim,
                weights=[word_embedding_matrix],
                trainable=False
            )
            self.position_embedding_layer = Embedding(
                input_dim=sequence_length, output_dim=output_dim,
                weights=[position_embedding_matrix],
                trainable=False
            )
                
        def get_position_encoding(self, seq_len, d, n=10000):
            P = np.zeros((seq_len, d))
            for k in range(seq_len):
                for i in np.arange(int(d/2)):
                    denominator = np.power(n, 2*i/d)
                    P[k, 2*i] = np.sin(k/denominator)
                    P[k, 2*i+1] = np.cos(k/denominator)
            return P

        def call(self, inputs):        
            position_indices = tf.range(tf.shape(inputs)[-1])
            embedded_words = self.word_embedding_layer(inputs)
            embedded_indices = self.position_embedding_layer(position_indices)
            return embedded_words + embedded_indices
else:
    def positional_encoding_numpy(d, ijk_3D, n=10000):
        # shape = (ijk_3D.shape[0], ijk_3D.shape[1], ijk_3D.shape[2], d)
        # P = np.zeros(shape)
        # for i_ind in range(ijk_3D.shape[0]):
        #     for j_ind in range(ijk_3D.shape[1]):
        #         for k_ind in range(ijk_3D.shape[2]):
        #             for d_ind in np.arange(int(d/2)):
        #                 i, j, k = ijk_3D[i_ind, j_ind, k_ind]
        #                 denominator = np.power(n, 2 * d_ind / d)
        #                 P[i, j, k, 2 * d_ind] = np.sin(i / denominator) + np.sin(j / denominator) + np.sin(k / denominator)
        #                 P[i, j, k, 2 * d_ind + 1] = np.cos(i / denominator) + np.cos(j / denominator) + np.cos(k / denominator)
        
        shape = (ijk_3D.shape[0], ijk_3D.shape[1], ijk_3D.shape[2], d)
        P = np.zeros(shape)
        for d_ind in np.arange(int(d/2)):
            # denominator_arr[:, :, :, d_ind] = np.power(n, 2 * d_ind / d)
            denominator = np.power(n, 2 * d_ind / d)
            P[:, :, :, 2 * d_ind] = np.sin(ijk_3D[:, :, :, 0] / denominator) + np.sin(ijk_3D[:, :, :, 1] / denominator) + np.sin(ijk_3D[:, :, :, 2] / denominator)
            P[:, :, :, 2 * d_ind + 1] = np.cos(ijk_3D[:, :, :, 0] / denominator) + np.cos(ijk_3D[:, :, :, 1] / denominator) + np.cos(ijk_3D[:, :, :, 2] / denominator)
        if (d % 2) == 1:
            denominator = np.power(n, 1)
            P[:, :, :, -1] = np.cos(ijk_3D[:, :, :, 0] / denominator) + np.cos(ijk_3D[:, :, :, 1] / denominator) + np.cos(ijk_3D[:, :, :, 2] / denominator)
        return P

class TimeDense(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(TimeDense, self).__init__()

        self.dense = Dense(*args, **kwargs)
        self.concat = Concatenate(axis = -1)
    
    def call(self, inputs, RE_DATE):
        return self.dense(self.concat([inputs, RE_DATE]))

class MultiDense(layers.Layer):
    def __init__(self, num_outputs, *args, **kwargs):
        super(MultiDense, self).__init__()

        self.num_outputs = num_outputs
        self.denses = [Dense(*args, **kwargs) for i in range(num_outputs)]
        # self.stack = Concatenate(axis = 1)
    
    def call(self, inputs):
        """
        inputs shape = (batch, time step, num features input)
        outputs shape = (batch, self.num_outputs, num features output)
        """

        outputs = []
        for idx_out in range(self.num_outputs):
            out_single_cluster = self.denses[idx_out](inputs) ## outputs shape: batch, time step, num features new
            out_single_cluster = tf.reduce_mean(out_single_cluster, axis= 1) ## outputs shape: batch, num features new
            outputs.append(out_single_cluster)
        return tf.stack(outputs, axis= 1) ## batch, self.num_outputs, num features new

class TransformerTemporal(layers.Layer):
    def __init__(self, input_dim, model_kwargs_attention = None, model_kwargs_prior_residual = None, model_kwargs_temporal = None, model_kwargs_post_residual = None, key_model_kwargs = None, query_model_kwargs = None, value_model_kwargs = None, prior_post_layer_norm_kwargs = None, droprate_before_residual= None, activation = "leaky_relu"):
        super(TransformerTemporal, self).__init__()

        ## Set default models.
        # assert(model_kwargs_attention[0] == "MultiHeadAttention")
        if model_kwargs_attention is None: self.model_kwargs_attention = ["MultiHeadAttention", dict(num_heads= 2, key_dim= 30)]
        else: self.model_kwargs_attention = deepcopy(model_kwargs_attention)

        if key_model_kwargs: self.key_model_kwargs = key_model_kwargs
        else: self.key_model_kwargs = dict(model= "Dense", kwargs= dict(units= 30, activation= activation))

        if query_model_kwargs: self.query_model_kwargs = query_model_kwargs
        else: self.query_model_kwargs = dict(model= "Dense", kwargs= dict(units= 30, activation= activation))

        if value_model_kwargs: self.value_model_kwargs = value_model_kwargs
        else: self.value_model_kwargs = dict(model= "Dense", kwargs= dict(units= 30, activation= activation))

        for model_kwargs in [self.key_model_kwargs, self.query_model_kwargs, self.value_model_kwargs]:
            if model_kwargs["model"] in ["LSTM", "GRU", "SimpleRNN"]:
                model_kwargs["kwargs"]["return_sequences"] = True
                model_kwargs["kwargs"]["return_state"] = True

        if model_kwargs_prior_residual is None: self.model_kwargs_prior_residual = []
        else: self.model_kwargs_prior_residual = deepcopy(model_kwargs_prior_residual)

        self.model_kwargs_temporal = deepcopy(model_kwargs_temporal)

        if model_kwargs_post_residual is None: self.model_kwargs_post_residual = []
        else: self.model_kwargs_post_residual = deepcopy(model_kwargs_post_residual)

        if prior_post_layer_norm_kwargs is not None: self.prior_post_layer_norm_kwargs = deepcopy(prior_post_layer_norm_kwargs)
        else: self.prior_post_layer_norm_kwargs = dict(prior= None, post= None)

        ## Build model
        self.key_model = keras_layer_dict[self.key_model_kwargs["model"]](**self.key_model_kwargs["kwargs"])
        self.query_model = keras_layer_dict[self.query_model_kwargs["model"]](**self.query_model_kwargs["kwargs"])
        self.value_model = keras_layer_dict[self.value_model_kwargs["model"]](**self.value_model_kwargs["kwargs"])
        self.att = keras_layer_dict[self.model_kwargs_attention[0]](**self.model_kwargs_attention[1])
        sequential = [keras_layer_dict[self.model_kwargs_prior_residual[i][0]](**self.model_kwargs_prior_residual[i][1]) for i in range(len(self.model_kwargs_prior_residual))]
        if self.prior_post_layer_norm_kwargs["prior"]: sequential.append(Dense(units = input_dim)) ## To match the dimension of input
        if droprate_before_residual is not None: sequential.append(Dropout(rate= droprate_before_residual))
        if len(sequential) > 0: self.prior_residual = Sequential(
            sequential
            )
        if self.model_kwargs_temporal: 
            self.temporal = keras_layer_dict[self.model_kwargs_temporal[0]](return_sequences= True, return_state= True, **self.model_kwargs_temporal[1])
        
        sequential = [keras_layer_dict[self.model_kwargs_post_residual[i][0]](**self.model_kwargs_post_residual[i][1]) for i in range(len(self.model_kwargs_post_residual))]
        if self.prior_post_layer_norm_kwargs["post"]:
            sequential.append(Dense(units = input_dim, activation= activation)) ## To match the dimension
        if droprate_before_residual is not None: sequential.append(Dropout(rate= droprate_before_residual))
        if len(sequential) > 0: self.post_residual = Sequential(sequential)

        if self.prior_post_layer_norm_kwargs["prior"] is not None: self.layer_norm_prior_residual = LayerNormalization(**self.prior_post_layer_norm_kwargs["prior"])
        if self.prior_post_layer_norm_kwargs["post"] is not None: self.layer_norm_post_residual = LayerNormalization(**self.prior_post_layer_norm_kwargs["post"])
    
    def call(self, inputs):
        """Sequence to sequence.
        
        Parameters
        ----------
        inputs: 
            shape is [batch, time step, num features].

        Return
        ------
        shape is [batch, time step, num features from self.post_residual or self.prior_residual or self.att (key_dim of self.model_kwargs_attention)]. -> No output dims are always same as input dim, if you want to adjust output dim, adds Dense layer after transformer.
        """

        outputs = inputs
        if self.key_model_kwargs["model"] in ["LSTM", "GRU", "SimpleRNN"]:
            key, _, _ = self.key_model(inputs)
        else:
            key = self.key_model(inputs)
        if self.query_model_kwargs["model"] in ["LSTM", "GRU", "SimpleRNN"]:
            query, _, _ = self.query_model(inputs)
        else:
            query = self.query_model(inputs)
        if self.value_model_kwargs["model"] in ["LSTM", "GRU", "SimpleRNN"]:
            value, _, _ = self.value_model(inputs)
        else:
            value = self.value_model(inputs)
        if self.model_kwargs_temporal: 
            outputs, vectors_last_hidden_state, vectors_last_cell_state  = self.temporal(outputs)
        # positional_encoding = 
        outputs = self.att(query = query, value = value, key = key)
        if len(self.model_kwargs_prior_residual) > 0 or self.prior_post_layer_norm_kwargs["prior"] is not None: 
            outputs = self.prior_residual(outputs)
        if self.prior_post_layer_norm_kwargs["prior"] is not None: 
            outputs = self.layer_norm_prior_residual(outputs + inputs)
        outputs_residual = outputs
        if len(self.model_kwargs_post_residual) > 0 or self.prior_post_layer_norm_kwargs["post"] is not None:
            outputs = self.post_residual(outputs)
            outputs = self.layer_norm_post_residual(outputs + outputs_residual)
        return outputs

def get_random_positions(num_batches, inner_shape, outer_shape):
    out = []
    for dim in range(len(inner_shape)): assert(inner_shape[dim] <= outer_shape[dim])
    for i in range(num_batches):
        position = []
        for dim in range(len(inner_shape)): 
            axis_position = randint(0, outer_shape[dim] - inner_shape[dim])
            position.append([axis_position, axis_position + inner_shape[dim]])
        out.append(position)
    return out

keras_layer_dict["TransformerTemporal"] = TransformerTemporal
keras_layer_dict["MultiDense"] = MultiDense
# keras_layer_dict["TimeDense"] = TimeDense

def get_batch_nparr(arr_xyzf, ijks_list):
    out = []
    for bi in range(len(ijks_list)):
        out.append(arr_xyzf[ijks_list[bi][0], ijks_list[bi][1], ijks_list[bi][2], :])
    return np.array(out)

# dense_kernel_regularizers = regularizers.l2(0.01)
# dense_kernel_regularizers = sparse_group_lasso
dense_kernel_regularizers = None

class GACC():
    def __init__(self, num_clusters = 50, min_max_scale_range : list = None, noise : str = 'uniform', kind_gen_loss_dis_entrybyentry = 'cross_entropy', weights_dict= None, small_delta : float = 1e-32, verbose = True, debug = True, run_eagerly= True):
        ## Sanity check for parameters.
        #added option for multi_modal distribution
        assert(noise in ['uniform', 'normal', 'multi_modal'])
        assert(small_delta < 1e-6)

        ## Set the attributes
        self.run_eagerly= run_eagerly

        if min_max_scale_range is None:
            print("WARNING: min_max_scaling is turned off.")
        else:
            assert(len(min_max_scale_range) == 2 and min_max_scale_range[0] < min_max_scale_range[1])
        self.min_max_scale_range = deepcopy(min_max_scale_range)
        self.num_padded = [[0, 0], [0, 0], [0, 0]]

        self.num_clusters = num_clusters
        self.noise = noise
        self.kind_gen_loss_dis_entrybyentry = kind_gen_loss_dis_entrybyentry
        if weights_dict is None:
            self.weights_dict = {"generator": {"discriminator": 1.0, "coop_x_to_c": 1.0, "coop_c_to_x": 1.0},}
        self.small_delta = small_delta
        self.verbose = verbose
        self.loss_dict = {} ## List of losses, which may be plotted to see the convergency.
        self.trained = False
        self.debug = debug
    
    def forward(self, model_name, input_loc):
        output = input_loc
        for tup, idx in zip(self.models_structure[model_name], range(len(self.models_structure[model_name]))):
            additional_inputs_for_model_init_local = {}
            if tup[0] == "TransformerTemporal":
                additional_inputs_for_model_init_local["input_dim"] = output.shape[-1]
            output = keras_layer_dict[tup[0]](**tup[1], **additional_inputs_for_model_init_local)(output)
        return output

    def set_trainable(self, train_model_name):
        self.models_dict[train_model_name].trainable = True
        for model_name in self.models_dict.keys():
            if model_name != train_model_name:
                self.models_dict[model_name].trainable = False
                # for i,layer in enumerate(self.models_dict[model_name].layers):
                #     layer.trainable = False
        if train_model_name in ["generator"]: self.models_dict[train_model_name].trainable = True

    def fit(self, data_xyzt, cluster_xyzp, num_points_sample = None, models_structure= None, optimizer = 'adam', mask_probs = None, epochs = 2, shuffle = True, mb_size = 32, num_batches = None, learning_rate_dicts = None, training_skip_prob = None, do_reweight = False, do_sample_random_position = True, additional_kwargs = None):
        for axis in [0, 1, 2]:
            assert(data_xyzt.shape[axis] == cluster_xyzp.shape[axis])
        assert(self.num_clusters >= cluster_xyzp.shape[3])
        self.dim_data = data_xyzt.shape[3]
        self.dim_latent = 30
        self.num_clusters_gt = cluster_xyzp.shape[3]
        num_samples = data_xyzt.shape[0] * data_xyzt.shape[1] * data_xyzt.shape[2]
        cluster_xyzp_scaled = utilsforminds.helpers.min_max_scale(cluster_xyzp, vmin = 0., vmax = 1., arr_min = cluster_xyzp.min(), arr_max = cluster_xyzp.max())
        if self.min_max_scale_range is not None:
            data_xyzt_scaled = utilsforminds.helpers.min_max_scale(data_xyzt, vmin = self.min_max_scale_range[0], vmax = self.min_max_scale_range[1], arr_min = data_xyzt.min(), arr_max = data_xyzt.max())

        num_points_sample = [8] if num_points_sample is None else num_points_sample

        if mask_probs is None: mask_probs = {"data_series": 0.5, "cluster_gt_data": 0.5}
        training_skip_prob_fit = utilsforminds.containers.merge_dictionaries([{'discriminator': 0., 'generator':0., "coop_x_to_c": 0, "coop_c_to_x": 0}, training_skip_prob])

        if learning_rate_dicts is None:
            self.learning_rate_dicts = {"generator_combined": 1e-3, "discriminator": 1e-3, "coop_x_to_c": 1e-3, "coop_c_to_x": 1e-3}
        else:
            self.learning_rate_dicts = deepcopy(learning_rate_dicts)
        if models_structure is None:
            self.models_structure = {}
            self.models_structure["generator"] = [["TransformerTemporal", dict()], ["TransformerTemporal", dict()]]
            self.models_structure["generator_cluster"] = [["MultiDense", dict(units= self.dim_latent, num_outputs= self.num_clusters, activation= "leaky_relu")]]
            self.models_structure["generator_data"] = [["TransformerTemporal", dict(model_kwargs_attention = ["MultiHeadAttention", dict(num_heads= 2, key_dim= 20)])], ["Dense", dict(units= self.dim_latent, activation= "leaky_relu")]]
            self.models_structure["predicted_clusters"] = [["Dense", dict(units= self.num_clusters, activation= 'sigmoid')]]
            # self.models_structure["discriminator_prior"] = [["TransformerTemporal", dict()], ["MultiDense", dict(units= 30, num_outputs= 1, activation= "leaky_relu")]]
            # self.models_structure["discriminator_post"] = [["Dense", dict(units= 30, activation= "leaky_relu")], ["Dense", dict(units= 1, activation= 'sigmoid')]]
            self.models_structure["discriminator"] = [["TransformerTemporal", dict()], ["MultiDense", dict(units= 30, num_outputs= 1, activation= "leaky_relu")], ["Dense", dict(units= 1, activation= 'sigmoid')]]
            self.models_structure["coop_x_to_c"] = [["TransformerTemporal", dict(model_kwargs_attention = ["MultiHeadAttention", dict(num_heads= 2, key_dim= 20)])], ["Dense", dict(units= 1, activation= 'sigmoid')]]
            self.models_structure["coop_c_to_x"] = [["TransformerTemporal", dict(model_kwargs_attention = ["MultiHeadAttention", dict(num_heads= 2, key_dim= 20)])], ["Dense", dict(units= self.dim_data, activation= "leaky_relu")]]
        else:
            self.models_structure = deepcopy(models_structure)

        # assert(self.models_structure["discriminator_prior"][-1][0] == "MultiDense")
        ## Starts new fresh session, while deleting possibly existing previous session.
        backend.clear_session()

        ijk_3D = np.transpose(np.stack(np.meshgrid(np.arange(data_xyzt_scaled.shape[0]), np.arange(data_xyzt_scaled.shape[1]), np.arange(data_xyzt_scaled.shape[2])), axis= 3), axes= [1, 0, 2, 3])
        assert(np.array_equal(ijk_3D[3, 7, 5], np.array([3, 7, 5])))
        # np.stack(np.meshgrid(np.arange(data_xyzt_scaled.shape[0]), np.arange(data_xyzt.shape[1]), np.arange(data_xyzt.shape[2])), axis= -1)
        data_xyzt_scaled = data_xyzt_scaled + positional_encoding_numpy(d= data_xyzt_scaled.shape[3], ijk_3D= ijk_3D)
        ## Set the vector length of cube.
        
        #%% Build Keras models.
        self.models_dict = {} ## Dict for models.
        self.inputs_dict = {"data_series": Input(shape = [None, data_xyzt_scaled.shape[3]], name="data_series"), 
                            "mask_series": Input(shape = [None, data_xyzt_scaled.shape[3]], name="mask_series"),
                            "cluster_gt_data": Input(shape = [None, self.num_clusters_gt], name="cluster_gt_data"),
                            "cluster_mask": Input(shape = [None, self.num_clusters], name="cluster_mask"),
                            "data_series_with_cluster": Input(shape = [None, data_xyzt_scaled.shape[3] + 1], name="data_series_with_cluster"),
                            "gen_real_01": Input(shape = (1), name="gen_real_01"),
                            "data_series_with_partial_mask_cluster": Input(shape = [None, data_xyzt_scaled.shape[3] + 2], name="data_series_with_partial_mask_cluster"),
                            "cluster_each_gt_data": Input(shape = [None], name="cluster_each_gt_data"),
                            "partial_data_series_with_mask_cluster": Input(shape = [None, data_xyzt_scaled.shape[3] * 2 + 1], name="partial_data_series_with_mask_cluster"),
                            # "data_each_gt_data": Input(shape = [None], name="data_each_gt_data"),
                            # "data_each_gt_mask": Input(shape = [None], name="data_each_gt_mask"),
                            # "cluster_gt_mask": Input(shape = [None, self.num_clusters_gt], name="cluster_gt_mask"),
                            # "mask_series_stack": Input(shape = [self.num_clusters, None, data_xyzt_scaled.shape[3]], name="mask_series_stack"),
                            # "mask_clusters": Input(shape = [None, self.num_clusters, 1], name="mask_clusters"),
                            }
        # batch_size = tf.shape(self.inputs_dict["data_series"])[0]
        self.outputs_dict = {"discriminator": []}
        self.models_outputs_from_gen_dict = {key: [] for key in ["discriminator", "coop_x_to_c", "coop_c_to_x"]}
        
        #%% Build Models.

        self.outputs_dict["discriminator"] = self.forward(model_name= "discriminator", input_loc= self.inputs_dict["data_series_with_cluster"])[:, 0, :]  ## shape: (batch, 1, 1)[:, 0, :]
        self.models_dict["discriminator"] = Model(inputs= [self.inputs_dict["data_series_with_cluster"], self.inputs_dict["gen_real_01"]], outputs = [self.outputs_dict["discriminator"]])
        self.models_dict['discriminator'].add_loss(self.binary_prob_loss(true_prob= self.inputs_dict["gen_real_01"], pred_prob= self.outputs_dict["discriminator"], axes= [1]))# loss_dim = (batch, )
        self.models_dict['discriminator'].add_metric(self.binary_prob_loss(true_prob= self.inputs_dict["gen_real_01"], pred_prob= self.outputs_dict["discriminator"], axes= [1]), name = "discriminator_loss")
        
        self.outputs_dict["coop_x_to_c"] = self.forward(model_name= "coop_x_to_c", input_loc= self.
                                                        inputs_dict["data_series_with_partial_mask_cluster"])  ## output shape: (batch, self.num_data, 1)
        self.models_dict["coop_x_to_c"] = Model(inputs= [self.inputs_dict["data_series_with_partial_mask_cluster"], self.inputs_dict["cluster_each_gt_data"]], outputs = [self.outputs_dict["coop_x_to_c"][:, :, 0]]) ## output shape: (batch, self.num_data, 1)[:, :, 0]]
        self.models_dict['coop_x_to_c'].add_loss(self.binary_prob_loss(true_prob= self.inputs_dict["cluster_each_gt_data"], pred_prob= self.outputs_dict["coop_x_to_c"][:, :, 0], axes= [1])) ## loss_dic = (batch, )
        self.models_dict['coop_x_to_c'].add_metric(self.binary_prob_loss(true_prob= self.inputs_dict["cluster_each_gt_data"], pred_prob= self.outputs_dict["coop_x_to_c"][:, :, 0], axes= [1]), name = "coop_x_to_c_loss")

        self.outputs_dict["coop_c_to_x"] = []
        self.outputs_dict["coop_c_to_x"] = self.forward(model_name= "coop_c_to_x", input_loc= self.
                                                        inputs_dict["partial_data_series_with_mask_cluster"]) ## (batch, self.num_data, dim_data)
        self.models_dict["coop_c_to_x"] = Model(inputs= [self.inputs_dict[key] for key in 
                                                        ["partial_data_series_with_mask_cluster", "data_series", "mask_series"]], outputs = [self.outputs_dict["coop_c_to_x"]])
        self.models_dict['coop_c_to_x'].add_loss(self.mse_loss(mask = self.
                                                        inputs_dict["mask_series"], true = self.
                                                        inputs_dict["data_series"], pred = self.outputs_dict["coop_c_to_x"], axes= [1, 2])) ## loss_dim = (batch, )
        self.models_dict['coop_c_to_x'].add_metric(self.mse_loss(mask = self.
                                                        inputs_dict["mask_series"], true = self.
                                                        inputs_dict["data_series"], pred = self.outputs_dict["coop_c_to_x"], axes= [1, 2]), name = "coop_c_to_x_loss")

        for model_name in ["discriminator", "coop_x_to_c", "coop_c_to_x"]:
            plot_model(self.models_dict[model_name], to_file=f'out/demo_{model_name}.pdf', show_shapes=True)
            self.models_dict[model_name].run_eagerly = self.run_eagerly ## has not great effect in debugging, self.model.compile(optimizer= optimizer, run_eagerly= run_eagerly) is more important for eager mode.
            if optimizer == 'adam':
                self.models_dict[model_name].compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= self.learning_rate_dicts[model_name]), run_eagerly= self.run_eagerly) ## .trainable attribute only affect BEFORE .compile model.
            else:
                raise Exception("Unsupported optimizer")
            if self.verbose:
                self.models_dict[model_name].summary()
                for i,layer in enumerate(self.models_dict[model_name].layers):
                    print(i,layer.name,layer.trainable)
            self.models_dict[model_name].trainable = False

        self.outputs_dict["generator"] = self.forward(model_name= "generator", input_loc= self.inputs_dict["data_series"]) ## shape: (batch, num_data, self.dim_latent)

        self.outputs_dict["generator_cluster"] = self.forward(model_name= "generator_cluster", input_loc= self.outputs_dict["generator"]) ## shape: (batch, self.num_clusters, self.dim_latent)
        
        self.outputs_dict["generator_data"] = self.forward(model_name= "generator_data", input_loc= self.outputs_dict["generator"])  ## shape: (batch, num_data, self.dim_latent)

        self.outputs_dict["predicted_clusters"] = self.forward(model_name= "predicted_clusters", input_loc= keras.layers.Dot(axes= (2, 2))([self.outputs_dict["generator_data"], self.outputs_dict["generator_cluster"]])) ## input shape: (batch, self.num_data, self.num_clusters), output shape: (batch, self.num_data, self.num_clusters)

        self.models_dict["generator"] = Model(inputs= [self.inputs_dict["data_series"]], outputs = [self.outputs_dict["generator_cluster"], self.outputs_dict["generator_data"], self.outputs_dict["predicted_clusters"]])

        discriminator_loss = 0.
        coop_x_to_c_loss = 0.
        coop_c_to_x_loss = 0.
        
        data_series = self.inputs_dict["data_series"]
        generator_cluster, generator_data, predicted_clusters = self.models_dict["generator"](self.inputs_dict["data_series"])
        predicted_clusters = tf.expand_dims(predicted_clusters, axis= -1) ## output shape: (batch, self.num_data, self.num_clusters, 1)

        for ci in range(self.num_clusters):
            data_series_with_cluster = tf.concat([data_series, predicted_clusters[:, :, ci, :]], axis= -1) ## output shape: (batch, self.num_data, self.dim_data + 1)
            # dummy_tensor = tf.fill(tf.shape(self.inputs_dict["gen_real_01"]), 99.0)

            dis_output = self.models_dict["discriminator"]({"data_series_with_cluster": data_series_with_cluster, "gen_real_01": tf.fill((tf.shape(self.inputs_dict["data_series"])[0], 1), 99.0)}) ## , "gen_real_01": dummy_tensor, output shape: (batch, 1, 1)[:, 0, :]
            self.models_outputs_from_gen_dict["discriminator"].append(dis_output)
            discriminator_loss = discriminator_loss + self.weights_dict["generator"]["discriminator"] * self.gen_prob_loss(pred_mask= dis_output, axes= [1])
            
            cluster_mask_expanded = tf.expand_dims(self.inputs_dict["cluster_mask"][:, :, ci], axis= -1) ## output shape: (batch, num_data, 1)
            data_series_with_cluster_masked = tf.concat([data_series, predicted_clusters[:, :, ci, :] * cluster_mask_expanded, cluster_mask_expanded], axis= -1) ## output shape: (batch, num_data, dim_data + 2)
            coop_x_to_c_output = self.models_dict["coop_x_to_c"]({"data_series_with_partial_mask_cluster": data_series_with_cluster_masked, "cluster_each_gt_data": predicted_clusters[:, :, ci, 0]}) ## output shape: (batch, self.num_data)
            self.models_outputs_from_gen_dict["coop_x_to_c"].append(coop_x_to_c_output)
            coop_x_to_c_loss = coop_x_to_c_loss + self.weights_dict["generator"]["coop_x_to_c"] * self.binary_prob_loss(true_prob= predicted_clusters[:, :, ci, 0], pred_prob= coop_x_to_c_output, axes= [1]) # loss_dim = (batch, )

            data_series_with_cluster_masked = tf.concat([data_series * self.inputs_dict["mask_series"], self.inputs_dict["mask_series"], predicted_clusters[:, :, ci, :]], axis= -1) ## output shape: (batch, self.num_data, 2 * dim_data + 1)
            coop_c_to_x_output = self.models_dict["coop_c_to_x"]({"partial_data_series_with_mask_cluster": data_series_with_cluster_masked, **{key: self.inputs_dict[key] for key in ["data_series", "mask_series"]}}) ## (batch, self.num_data, dim_data)
            self.models_outputs_from_gen_dict["coop_c_to_x"].append(coop_c_to_x_output)
            coop_c_to_x_loss = coop_c_to_x_loss + self.weights_dict["generator"]["coop_c_to_x"] * self.mse_loss(mask = self.inputs_dict["mask_series"], true = self.inputs_dict["data_series"], pred = coop_c_to_x_output, axes= [1, 2]) ## loss_dim = (batch, )
        
        # inputs_dict_generator_combined = self.inputs_dict["generator_combined"]
        # inputs_dict_generator_combined["data_each_gt_mask"] = self.inputs_dict["coop_c_to_x"]["data_each_gt_mask"]
        # inputs_dict_generator_combined["data_each_gt_data"] = self.inputs_dict["coop_c_to_x"]["data_each_gt_data"]
        self.models_dict["generator_combined"] = Model(inputs= [self.inputs_dict[key] for key in 
                                                        ["data_series", "cluster_gt_data", "cluster_mask", "mask_series"]], outputs = [self.models_outputs_from_gen_dict[model_name][ci] for ci in range(self.num_clusters) for model_name in ["discriminator", "coop_x_to_c", "coop_c_to_x"]])

        self.models_dict['generator_combined'].add_loss(self.binary_prob_loss(true_prob = self.inputs_dict["cluster_gt_data"], pred_prob = predicted_clusters[:, :, :self.num_clusters_gt, 0], loss_mask= self.inputs_dict["cluster_mask"][:, :, :self.num_clusters_gt], axes= [1, 2]))
        self.models_dict['generator_combined'].add_metric(self.binary_prob_loss(true_prob = self.inputs_dict["cluster_gt_data"], pred_prob = predicted_clusters[:, :, :self.num_clusters_gt, 0], loss_mask= self.inputs_dict["cluster_mask"][:, :, :self.num_clusters_gt], axes= [1, 2]), name= "generator_cluster_gt")

        for loss, loss_name in zip([discriminator_loss, coop_x_to_c_loss, coop_c_to_x_loss], ["gen_discriminator_loss", "gen_coop_x_to_c_loss", "gen_coop_c_to_x_loss"]):
            self.models_dict['generator_combined'].add_loss(loss)
            self.models_dict['generator_combined'].add_metric(loss, name= loss_name)
        
        plot_model(self.models_dict["generator_combined"], to_file=f'out/demo_generator_combined.pdf', show_shapes=True)
        self.models_dict["generator_combined"].run_eagerly = self.run_eagerly ## has not great effect in debugging, self.model.compile(optimizer= optimizer, run_eagerly= run_eagerly) is more important for eager mode.
        if optimizer == 'adam':
            self.models_dict["generator_combined"].compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= self.learning_rate_dicts["generator_combined"]), run_eagerly= self.run_eagerly) ## .trainable attribute only affect BEFORE .compile model.
        else:
            raise Exception("Unsupported optimizer")
        if self.verbose:
            self.models_dict["generator_combined"].summary()
            for i,layer in enumerate(self.models_dict["generator_combined"].layers):
                print(i,layer.name,layer.trainable)
        
        if False:
            true_mask_1 = tf.fill((tf.shape(self.outputs_dict["discriminator_post"])[0], self.num_clusters_gt, 1), 1.0) ## shape: (batch, self.num_clusters_gt, 1)
            true_mask_0 = tf.fill((tf.shape(self.outputs_dict["discriminator_post"])[0], self.num_clusters - self.num_clusters_gt, 1), 0.0)
            true_mask = tf.concat([true_mask_1, true_mask_0], axis= 1)

            self.models_dict['generator'].add_loss(self.weights_dict["generator"]["discriminator"] * self.gen_prob_loss(true_mask= true_mask, pred_mask= self.outputs_dict["discriminator_post"], axes= [1, 2]) 
            + self.weights_dict["generator"]["coop_x_to_c"] * self.binary_prob_loss(true_prob= self.outputs_dict["predicted_clusters"], pred_prob= self.outputs_dict["coop_x_to_c"], axes= [1, 2])
            + self.weights_dict["generator"]["coop_c_to_x"] * self.mse_loss(mask = self.outputs_dict["mask_series_stack"], true = self.outputs_dict["data_series_stack"], pred = self.outputs_dict["coop_c_to_x"], axes= [1, 2]))

            self.models_dict['generator'].add_metric(self.weights_dict["generator"]["discriminator"] * self.gen_prob_loss(true_mask= true_mask, pred_mask= self.outputs_dict["discriminator_post"], axes= [1, 2]) 
            + self.weights_dict["generator"]["coop_x_to_c"] * self.binary_prob_loss(true_prob= self.outputs_dict["predicted_clusters"], pred_prob= self.outputs_dict["coop_x_to_c"], axes= [1, 2])
            + self.weights_dict["generator"]["coop_c_to_x"] * self.mse_loss(mask = self.outputs_dict["mask_series_stack"], true = self.outputs_dict["data_series_stack"], pred = self.outputs_dict["coop_c_to_x"], axes= [1, 2]), name = "generator_loss")

        ## Define Models

        ## Check freezing / training layers. Should be put before adding loss functions.
        models_layer_names = {model_name: [layer.name for i, layer in enumerate(self.models_dict[model_name].layers)] for model_name in self.models_dict.keys()}

        model_names_to_test = ["generator", "discriminator", "coop_x_to_c", "coop_c_to_x"]
        layers_test = {model_name: {"fix": None, "train": None} for model_name in model_names_to_test}
        for model_name_train in model_names_to_test:
            for i, layer in enumerate(self.models_dict[model_name_train].layers):
                # layer.trainable = True
                if len(layer.get_weights()) > 0 and ((True or model_name_train in ["generator"] or layer.name not in models_layer_names["generator"]) and layers_test[model_name_train]["train"] is None and random() < 0.05):
                    layers_test[model_name_train]["train"] = {"layer_name": layer.name, "weights": np.copy(layer.get_weights()[0])}
            for model_name_fix in model_names_to_test:
                if model_name_fix != model_name_train:
                    # self.models_dict[model_name_fix].trainable = False
                    for i, layer in enumerate(self.models_dict[model_name_fix].layers):
                        # layer.trainable = False
                        if len(layer.get_weights()) > 0 and ((True or model_name_train not in ["generator"] or layer.name not in models_layer_names["generator"]) and layers_test[model_name_train]["fix"] is None and random() < 0.05):
                            layers_test[model_name_train]["fix"] = {"model_name": model_name_fix, "layer_name": layer.name, "weights": np.copy(layer.get_weights()[0])}

        ## Freeze the generator. When you train something, you need to freeze the others, in GAN.
        ## https://stackoverflow.com/questions/56675964/what-is-the-difference-between-setting-a-keras-model-trainable-vs-making-each-la

        #%% Train model
        self.losses_dict = {}

        if epochs == "auto": 
            if not do_sample_random_position: epochs = int(96000 / num_samples) ## will take around 3 hours, 8000 for each hour.
            else: epochs = int(9600 * 10 / num_samples)

        locs_ijk = [list(range(data_xyzt_scaled.shape[0])), list(range(data_xyzt_scaled.shape[1])), list(range(data_xyzt_scaled.shape[2]))]
        print(f"Training with {epochs} epochs.")
        for it in range(epochs):
            ## Prepare mbIdcs which is the indices of samples of each batch.
            if not do_sample_random_position:
                raise NotImplementedError()
                if shuffle:
                    idc = list(range(num_samples))
                    np.random.shuffle(idc)
                    mbIdcs = helpers.splitList(False, mb_size, idc) ## Indices of samples of each batch.
                else:
                    idc = list(range(num_samples))
                    mbIdcs = helpers.splitList(False, mb_size, idc) ## Indices of samples of each batch.
                tqdm_obj = tqdm(mbIdcs)
            else:
                if num_batches is None: tqdm_obj = tqdm(range(num_samples // mb_size)) ## Total Samples // Batch Size
                else: tqdm_obj = tqdm(range(num_batches))
            
            for batch_idx, mbIdc in enumerate(tqdm_obj): # for each mini batch, mbIdc = [15, 201, 154, ...] is list of indices.
                if not do_sample_random_position: mbSizeCurr = len(mbIdc) # at the last portion, number of data remained may be smaller than mini batch size
                else: mbSizeCurr = mb_size
                ijks_list = [[list(sample(locs_ijk[axis], sample(num_points_sample, 1)[0])) for axis in range(3)] for bi in range(mbSizeCurr)]

                ### Prepare input data of mini batch.
                inputs_dict_loc = {"data_series": get_batch_nparr(data_xyzt_scaled, ijks_list= ijks_list), "cluster_gt_data": get_batch_nparr(cluster_xyzp_scaled, ijks_list= ijks_list)}
                inputs_dict_loc["mask_series"] = np.where(np.random.rand(*inputs_dict_loc["data_series"].shape) > mask_probs["data_series"], 1., 0.)
                inputs_dict_loc["cluster_mask"] = np.where(np.random.rand(inputs_dict_loc["cluster_gt_data"].shape[0], inputs_dict_loc["cluster_gt_data"].shape[1], self.num_clusters) > mask_probs["cluster_gt_data"], 1., 0.)

                #%% -- Train model --
                ## Freeze generator, while training discriminator.
                for model_name in self.models_dict.keys():
                    # self.set_trainable(model_name) ## .trainable attribute only affect BEFORE .compile model.
                    model_name_trained = "generator" if model_name == "generator_combined" else model_name

                    # for i, layer in enumerate(self.models_dict[model_name].layers):
                    #     layer.trainable = True
                    #     if len(layer.get_weights()) > 0 and (model_name in ["generator"] or layer.name not in models_layer_names["generator"] and layers_test["train"] is None and random() < 0.1):
                    #         layers_test["train"] = {"layer_name": layer.name, "weights": np.copy(layer.get_weights()[0])}

                    # for model_name_2 in self.models_dict.keys():
                    #     if model_name_2 != model_name:
                    #         self.models_dict[model_name_2].trainable = False
                    #         for i, layer in enumerate(self.models_dict[model_name_2].layers):
                    #             layer.trainable = False
                    #             if len(layer.get_weights()) > 0 and (model_name not in ["generator"] or layer.name not in models_layer_names["generator"] and layers_test["fix"] is None and random() < 0.1):
                    #                 layers_test["fix"] = {"model_name": model_name_2, "layer_name": layer.name, "weights": np.copy(layer.get_weights()[0])}

                    # if model_name in ["generator"]:
                    #     self.models_dict[model_name].trainable = True
                    #     for i, layer in enumerate(self.models_dict[model_name].layers):
                    #         layer.trainable = True
                
                    if random() > training_skip_prob_fit[model_name_trained]: # To ensure the weights of discriminator or generator do not change during the training phase of counterpart
                        if self.verbose > 0: print(f"Training {model_name}.")
                        if self.debug: # To ensure the weights of discriminator or generator do not change during the training phase of counterpart
                            if layers_test[model_name_trained]["fix"] is not None:
                                for layer in self.models_dict[layers_test[model_name_trained]["fix"]["model_name"]].layers:
                                    if layer.name == layers_test[model_name_trained]["fix"]["layer_name"]:
                                        layers_test[model_name_trained]["fix"]["weights"] = layer.get_weights()[0]
                                        break
                            if layers_test[model_name_trained]["train"] is not None:
                                for layer in self.models_dict[model_name_trained].layers:
                                    if layer.name == layers_test[model_name_trained]["train"]["layer_name"]:
                                        layers_test[model_name_trained]["train"]["weights"] = layer.get_weights()[0]
                                        break

                        if model_name in ["generator_combined"]: ## generator is already included
                            loss_this = self.models_dict[model_name].train_on_batch(x = inputs_dict_loc, y={}, return_dict=True)
                            # loss_this = {out: loss_this[i] for i, out in enumerate(self.models_dict[model_name].metrics_names)}
                        elif model_name in ["discriminator", "coop_c_to_x", "coop_x_to_c"]:
                            # generator_cluster, generator_data, predicted_clusters = self.models_dict["generator"](inputs_dict_loc["data_series"])
                            generator_cluster, generator_data, predicted_clusters = self.models_dict["generator"].predict(inputs_dict_loc["data_series"])
                            predicted_clusters = np.expand_dims(predicted_clusters, axis= -1) ## output shape: (batch, num_data, num_clusters, 1)
                            loss_this = {}
                            for ci in range(self.num_clusters):
                                cluster_predicted_data_ci = predicted_clusters[:, :, ci] ## output shape: (batch, num_data, 1)
                                cluster_mask_ci_expanded = np.expand_dims(inputs_dict_loc["cluster_mask"][:, :, ci], axis= -1) ## output shape: (batch, num_data, 1)
                                if model_name == "discriminator":
                                    if ci < self.num_clusters_gt:
                                        cluster_gt_data_ci = np.expand_dims(inputs_dict_loc["cluster_gt_data"][:, :, ci], axis = -1)
                                        data_series_with_cluster = np.concatenate([inputs_dict_loc["data_series"], cluster_gt_data_ci], axis= -1)
                                        gen_real_01 = np.array([[1.] for bi in range(mbSizeCurr)])
                                    else:
                                        data_series_with_cluster = np.concatenate([inputs_dict_loc["data_series"], cluster_predicted_data_ci], axis= -1)
                                        gen_real_01 = np.array([[0.] for bi in range(mbSizeCurr)])
                                    inputs_dict_loc_batch = {"data_series_with_cluster": data_series_with_cluster, "gen_real_01": gen_real_01}

                                elif model_name == "coop_c_to_x":
                                    partial_data_series_with_mask_cluster = np.concatenate([inputs_dict_loc["data_series"] * inputs_dict_loc["mask_series"], inputs_dict_loc["mask_series"], cluster_predicted_data_ci], axis= -1)
                                    inputs_dict_loc_batch = {"partial_data_series_with_mask_cluster": partial_data_series_with_mask_cluster, "data_series": inputs_dict_loc["data_series"], "mask_series": inputs_dict_loc["mask_series"]}
                                
                                elif model_name == "coop_x_to_c":
                                    data_series_with_partial_mask_cluster = np.concatenate([inputs_dict_loc["data_series"], cluster_predicted_data_ci * cluster_mask_ci_expanded, cluster_mask_ci_expanded], axis= -1)
                                    inputs_dict_loc_batch = {"data_series_with_partial_mask_cluster": data_series_with_partial_mask_cluster, "cluster_each_gt_data": cluster_predicted_data_ci[:, :, 0]}

                                loss_this_loc = self.models_dict[model_name].train_on_batch(x = inputs_dict_loc_batch, y={}, return_dict=True)
                                for key in loss_this_loc.keys():
                                    if key in loss_this.keys():
                                        loss_this[key] += loss_this_loc[key]
                                    else:
                                        loss_this[key] = loss_this_loc[key]

                        # if self.verbose: print(f"loss of {model_name}:\n\t {loss_this}.")
                        for key in loss_this.keys():
                            if key in self.loss_dict.keys():
                                self.loss_dict[key].append(loss_this[key])
                            else:
                                self.loss_dict[key] = [loss_this[key]]

                        if self.debug: # To ensure the weights of discriminator or generator do not change during the training phase of counterpart
                            # if self.verbose > 0: print(f"Existence of fix layer: {layers_test[model_name]['fix'] is not None} and train layer: {layers_test[model_name]['train'] is not None}.")
                            if layers_test[model_name_trained]["fix"] is not None:
                                for layer in self.models_dict[layers_test[model_name_trained]["fix"]["model_name"]].layers:
                                    if layer.name == layers_test[model_name_trained]["fix"]["layer_name"]:
                                        assert(np.allclose(layer.get_weights()[0], layers_test[model_name_trained]["fix"]["weights"]))
                                        break
                            if layers_test[model_name_trained]["train"] is not None:
                                for layer in self.models_dict[model_name_trained].layers:
                                    if layer.name == layers_test[model_name_trained]["train"]["layer_name"]:
                                        if (np.allclose(layer.get_weights()[0], layers_test[model_name_trained]["train"]["weights"], rtol = 0., atol = 0.)):
                                            print(f"WARNING: Weights of {layer.name} of {model_name_trained} are not changed after training on a batch")
                                        else:
                                            print(f"PASS: Weights of {layer.name} of {model_name_trained} are changed after training on a batch")
                                        break
                
                    if False and self.verbose:
                        for model_name_3 in self.models_dict.keys():
                            print(f"Learning Rate of {model_name_3}: {self.models_dict[model_name_3].optimizer._decayed_lr('float32').numpy()}")
                if self.verbose:
                    for key in self.loss_dict.keys():
                        print(f"loss of {key}:\n\t {self.loss_dict[key]}.")
                
                if do_reweight: ## Reweight the factor of each loss term. Decrease the factor of decreasing loss or Increase the factor of increasing loss.
                    base_factor = 0.1
                    reweight_verbose = False
                    self.weight_MSE_fit = utilsforminds.math.get_new_weight_based_loss_trends(losses = self.loss_dict["gen_loss_mse"], current_weight = self.weight_MSE_fit, factor_weight_change_to_loss_change = abs(self.weight_MSE * base_factor), max_weight = abs(self.weight_MSE * 1e+3), verbose = reweight_verbose)
                    if self.use_discriminator_entire: 
                        self.weight_of_dis_entire_density_for_gen_fit = utilsforminds.math.get_new_weight_based_loss_trends(losses = self.loss_dict["gen_loss_dis_entire"], current_weight = self.weight_of_dis_entire_density_for_gen_fit, factor_weight_change_to_loss_change = abs(self.weight_of_dis_entire_density_for_gen * base_factor), max_weight = abs(self.weight_of_dis_entire_density_for_gen * 1e+3), verbose = reweight_verbose)
                    if self.use_discriminator_entire: 
                        training_skip_prob_fit["discriminator_entire"] = utilsforminds.math.get_new_weight_based_loss_trends(losses = self.loss_dict["gen_loss_dis_entire"], current_weight = training_skip_prob_fit["discriminator_entire"], factor_weight_change_to_loss_change = abs(0.01), max_weight = 0.99, verbose = reweight_verbose)
                    self.weight_continuity_loss_fit = utilsforminds.math.get_new_weight_based_loss_trends(losses = self.loss_dict["gen_loss_continuity_with_schatten_pnorm"], current_weight = self.weight_continuity_loss_fit, factor_weight_change_to_loss_change = abs(self.weight_continuity_loss * base_factor), max_weight = abs(self.weight_continuity_loss * 1e+3), verbose = reweight_verbose)
                    self.weight_of_dis_loss_for_gen_fit = utilsforminds.math.get_new_weight_based_loss_trends(losses = self.loss_dict["gen_loss_dis_entrybyentry"], current_weight = self.weight_of_dis_loss_for_gen_fit, factor_weight_change_to_loss_change = abs(self.weight_of_dis_loss_for_gen * base_factor), max_weight = abs(self.weight_of_dis_loss_for_gen * 1e+3), verbose = reweight_verbose)
                    training_skip_prob_fit["discriminator"] = utilsforminds.math.get_new_weight_based_loss_trends(losses = self.loss_dict["gen_loss_dis_entrybyentry"], current_weight = training_skip_prob_fit["discriminator"], factor_weight_change_to_loss_change = abs(0.01), max_weight = 0.99, verbose = reweight_verbose)
                
                ## Convert metrics to dictionary with metric name: https://github.com/keras-team/keras/issues/14045
        print("Training finished.")
    
    def prepare_inputs_whole(self, data_xyzt, do_sample_random_position, predict_phase = False, cube_shape = None, cube_strides = None):
        result_dict = {}
        input_shape = deepcopy(data_xyzt.shape)
        mask_arr_tmp = np.ones(input_shape)
        ijk_3D = np.transpose(np.stack(np.meshgrid(np.arange(data_xyzt.shape[0]), np.arange(data_xyzt.shape[1]), np.arange(data_xyzt.shape[2])), axis= 3), axes= [1, 0, 2, 3])
        assert(np.array_equal(ijk_3D[5, 0, 2], np.array([5, 0, 2])))

        #%% min max scaling.
        if self.min_max_scale_range is not None:
            data_xyzt_tmp = data_xyzt.astype(np.float32) # shallow copy
        else:
            if False:
                data = []
                if not predict_phase:
                    self.name_min_max = [] ## Save the original range, to recover the original range in the future.
                
                raw_data = data_xyzt
                if predict_phase:
                    data = []
                    for channel in range(input_shape.shape[3]):
                        tmp = raw_data[:, :, :, channel]
                        data.append(utilsforminds.helpers.min_max_scale(tmp, vmin = self.min_max_scale_range[0], vmax = self.min_max_scale_range[1], arr_min = self.name_min_max[channel]['min'], arr_max = self.name_min_max[channel]['max']))
                    data_xyzt_tmp = np.stack(data, axis = 3).astype(np.float32) # deep copy
                else:
                    data = []
                    for channel in range(input_shape.shape[3]): ## Save original range for each channel.
                        tmp = raw_data[:, :, :, channel]
                        minCurr = tmp.min()
                        maxCurr = tmp.max()
                        data.append(utilsforminds.helpers.min_max_scale(arr = tmp, vmin = self.min_max_scale_range[0], vmax = self.min_max_scale_range[1], arr_min = minCurr, arr_max = maxCurr))
                        self.name_min_max.append({'min': minCurr, 'max': maxCurr})
                    data_xyzt_tmp = np.stack(data, axis = 3).astype(np.float32) # deep copy
                    if False and self.debug:
                        for channel in range(self.num_channels_dict[data_kind]): ## Check if recovered array has same value as original array before min-max scaling.
                            # assert(np.allclose(amount_arr[:, :, :, channel], helpers.min_max_scale(data_xyzt_tmp[:, :, :, channel], vmin = self.name_min_max[channel]['min'], vmax = self.name_min_max[channel]['max'], onlyPositive = False)))
                            assert(np.allclose(data_xyzt[:, :, :, channel], utilsforminds.helpers.min_max_scale(data_xyzt_tmp[:, :, :, channel], vmin = self.name_min_max[data_kind][channel]['min'], vmax = self.name_min_max[data_kind][channel]['max'], arr_min = self.min_max_scale_range[0], arr_max = self.min_max_scale_range[1]), rtol=1e-01, atol=1e-04))
                data = [] ## Free memory.
            else:
                data_xyzt_tmp = utilsforminds.helpers.min_max_scale(data_xyzt, vmin = self.min_max_scale_range[0], vmax = self.min_max_scale_range[1], arr_min = data_xyzt.min(), arr_max = data_xyzt.max())

        if predict_phase:
            result_dict["mask_arr_included"] = np.zeros(input_shape)
            idcMap = []
            ijk_3D_flat = []
        
        if False:
            #%% The surrounding space of whole space padded.
            ## pad array for convolution.
            if self.padding_dict['amount'] == 'constant_0':
                data_xyzt_tmp_padded = np.pad(data_xyzt_tmp, self.num_padded + [[0, 0]], mode = 'constant', constant_values = 0) ## [[0, 0]] is for channel dimension.
                if self.use_geophysics_hint: 
                    data_xyzt_tmp_hint_padded = np.pad(data_xyzt_tmp_hint, self.num_padded + [[0, 0]], mode = 'constant', constant_values = 0)
                    result_dict["data_xyzt_tmp_hint_padded"] = data_xyzt_tmp_hint_padded
            if self.padding_dict['counter'] == 'constant_0':
                counter_arr_tmp_padded = np.pad(counter_arr_tmp, self.num_padded + [[0, 0]], mode = 'constant', constant_values = 0)
                mask_arr_tmp_padded = np.pad(mask_arr_tmp, self.num_padded + [[0, 0]], mode = 'constant', constant_values = 0)
                if self.use_geophysics_hint: 
                    counter_arr_tmp_hint_padded = np.pad(counter_arr_tmp_hint, self.num_padded + [[0, 0]], mode = 'constant', constant_values = 0)
                    result_dict["counter_arr_tmp_hint_padded"] = counter_arr_tmp_hint_padded
        
        if not do_sample_random_position:
            ## Temporary list to stack cubes and shells.
            tmpAmountVec = []
            tmpMaskVec = []

            if cube_shape is not None:
                ## Calculate the indices of positions for each cube.
                idcForEachAxis = {0:[], 1:[], 2:[]}  # x, y, z directions, {0:[[0, 3], [2, 5], ..], 1:[[0, 6], [4, 10], ..], 2:[[2, 4], [3, 5], ..]}
                for axis in range(3): ## For each axis.
                    startIdx = 0
                    endIdx = cube_shape[axis]
                    while(True): ## Before exceeds.
                        if(endIdx >= input_shape[axis]):
                            idcForEachAxis[axis].append((input_shape[axis] - cube_shape[axis], input_shape[axis]))
                            break
                        idcForEachAxis[axis].append((startIdx, endIdx))
                        
                        startIdx = startIdx + cube_strides[axis]
                        endIdx = endIdx + cube_strides[axis]
                
                ## Iterate the calculated indices and stack the data(cube or shell).
                shell_idx_position_map = []
                for xIdc in idcForEachAxis[0]: ## x, y, z directions, idcForEachAxis= {0:[[0, 3], [2, 5], ..], 1:[[0, 6], [4, 10], ..], 2:[[2, 4], [3, 5], ..]}.
                    for yIdc in idcForEachAxis[1]:
                        for zIdc in idcForEachAxis[2]:
                            shell_idx_position_map.append([[xIdc[0], xIdc[1] + self.num_padded[0][0] + self.num_padded[0][1]], [yIdc[0], yIdc[1] + self.num_padded[1][0] + self.num_padded[1][1]], [zIdc[0], zIdc[1] + self.num_padded[2][0] + self.num_padded[2][1]]]) ## Ranges of shell size.
                            tmpMaskVec.append(mask_arr_tmp[xIdc[0]:xIdc[1], yIdc[0]:yIdc[1], zIdc[0]:zIdc[1], :].reshape(((xIdc[1] - xIdc[0]) * (yIdc[1] - yIdc[0]) * (zIdc[1] - zIdc[0]), input_shape[3]))) ## cube for dense. arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2]) 
                            tmpAmountVec.append(data_xyzt_tmp[xIdc[0]:xIdc[1], yIdc[0]:yIdc[1], zIdc[0]:zIdc[1], :].reshape(((xIdc[1] - xIdc[0]) * (yIdc[1] - yIdc[0]) * (zIdc[1] - zIdc[0]), input_shape[3]))) ## cube for dense.
                            ijk_3D_flat.append(ijk_3D[xIdc[0]:xIdc[1], yIdc[0]:yIdc[1], zIdc[0]:zIdc[1], :].reshape(((xIdc[1] - xIdc[0]) * (yIdc[1] - yIdc[0]) * (zIdc[1] - zIdc[0]), 3)))
                            if predict_phase:
                                result_dict["mask_arr_included"][xIdc[0]:xIdc[1], yIdc[0]:yIdc[1], zIdc[0]:zIdc[1], :] += np.where(result_dict["mask_arr_included"][xIdc[0]:xIdc[1], yIdc[0]:yIdc[1], zIdc[0]:zIdc[1], :] == 1., 0., np.where(mask_arr_tmp[xIdc[0]:xIdc[1], yIdc[0]:yIdc[1], zIdc[0]:zIdc[1], :] == 1., 1., 0.))
                            idcMap.append((xIdc, yIdc, zIdc))
            else:
                raise Exception(NotImplementedError)

            ## Stacked list of 3D arrays -> 4D array.
            num_samples = len(shell_idx_position_map)

            result_dict["shell_idx_position_map"] = np.array(shell_idx_position_map)
            result_dict["data_series_batches"] = np.stack(tmpAmountVec, axis = 0).astype(np.float32)
            result_dict["vector_cube_mask_total"] = np.stack(tmpMaskVec, axis = 0).astype(np.float32)
            result_dict["ijk_3D_flat"] = np.stack(ijk_3D_flat, axis = 0)

            ## Free memory.
            tmpAmountVec = []
            tmpMaskVec = []
        else:
            num_samples = input_shape[0] * input_shape[1] * input_shape[2] // (cube_strides[0] * cube_strides[1] * cube_strides[2])
        
        result_dict.update(dict(data_xyzt_tmp= data_xyzt_tmp, num_samples= num_samples))

        if predict_phase:
            result_dict.update(dict(idcMap= idcMap))
        return result_dict
    
    def predict(self, data_xyzt, num_points_sample= None, filterWithDiscriminator = False, getPopulatedMask = False, maskKeepThreshold = 0.7, maskKeepThresholdType = 'rank', keep_avg = False):
        """Predicts the sparse mineral distribution given by amount_arr and counter_arr.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """

        ## check the sanity of input.
        assert(len(data_xyzt.shape) == 4)
        assert(data_xyzt.shape[3] == self.dim_data)
        assert(maskKeepThresholdType in ['rank', 'absolute'])
        shape_cluster_xyzt = [data_xyzt.shape[axis] for axis in range(3)] + [self.num_clusters]
        cluster_xyzt_pred = np.zeros(shape_cluster_xyzt)
        overlappedArr = np.zeros(shape_cluster_xyzt)
        
        mb_size = 16

        if False:
            data_prep_whole_dict = self.prepare_inputs_whole(amount_arr = amount_arr, counter_arr = counter_arr, additional_kwargs= additional_kwargs, filter_num_observation_threshold_dict= filter_num_observation_threshold_dict, do_sample_random_position= False, predict_phase= True)

            amount_arr_scaled = data_prep_whole_dict["data_xyzt_tmp"]
            counterArrInput01 = data_prep_whole_dict["mask_arr_tmp"]
            idcMap = data_prep_whole_dict["idcMap"]
            mask_arr_included = data_prep_whole_dict["mask_arr_included"]
            counter_arr_copied = data_prep_whole_dict["counter_arr_tmp"]

            amount_arr_imputed = np.zeros(shapeInput) ## Imputed output.
            shell_idx_position_map = data_prep_whole_dict["shell_idx_position_map"] ## Save the positions (indices) of each cube. This is needed when recover the output array.

            ## Stacked list of 3D arrays -> 4D array.
            vector_cube_counter_total = data_prep_whole_dict["vector_cube_counter_total"]
            vector_cube_amount_total = data_prep_whole_dict["vector_cube_amount_total"]
            vector_cube_mask_total = data_prep_whole_dict["vector_cube_mask_total"]

            counter_arr_copied_padded = data_prep_whole_dict["counter_arr_tmp_padded"]
            amount_arr_scaled_padded = data_prep_whole_dict["data_xyzt_tmp_padded"]
        data_prep_whole_dict = self.prepare_inputs_whole(data_xyzt = data_xyzt, do_sample_random_position = False, predict_phase = True, cube_shape = [2, 2, 2], cube_strides = [2, 2, 2])

        # if filterWithDiscriminator or getPopulatedMask: ## To get the certainty distribution (output of discriminator).
        #     counter_arr_imputed = np.zeros(shapeInput)
        #     counter_arr_imputed_raw = np.zeros(shapeInput)

        
        ## Set the indices of samples of each batch.
        idcMap = data_prep_whole_dict["idcMap"]
        idc = list(range(len(idcMap)))
        mb_indices = helpers.splitList(False, mb_size, idc)
        
        for mb_indice in tqdm(mb_indices): ## For example, mb_indice = [15, 234, 56, ...]
            mb_size_curr = len(mb_indice) # at the last portion, number of data remained may be smaller than mini batch size
            ## Prepare data of mini batch
            ## Prepare vectors of cubes for dense layer.
            data_series = data_prep_whole_dict["data_series_batches"][mb_indice]

            ## Prepare 4D arrays of shells for convolutional layer.
            # shell_amount_batch = get_shells_batch(amount_arr_scaled_padded, shell_idx_position_map[mb_indice])
            # shell_mask_batch = np.where(shell_counter_batch >= 1., 1., 0.)

            ## Prepare noise vactor for dense layer and noise cube for convolutional layer.
            
            ## Predicts, important part.
            generator_cluster, generator_data, predicted_clusters = self.models_dict["generator"].predict(data_series) ## predicted_clusters shape, (batch, num_data, num_clusters)
            
            ### From the imputed cubes of subspace, we populate the whole space imputed.
            ## By putting the imputed cubes on the subspace, recover the small portion of array of whole space. 
            for idx_in_total, idx_in_mb in zip(mb_indice, range(len(mb_indice))):
                ## Counts the imputation count to avoid overwritting.
                for pstep in range(data_prep_whole_dict["ijk_3D_flat"][idx_in_total].shape[0]):
                    i, j, k = data_prep_whole_dict["ijk_3D_flat"][idx_in_total][pstep]
                    overlappedArr[i, j, k, :] += 1. ## We need to handle the overlapped imputations.
                    cluster_xyzt_pred[i, j, k, :] += predicted_clusters[idx_in_mb][pstep]
                    if False and self.debug: ## If recovering process was correct, then the input mineral amount should be preserved.
                        assert(np.allclose((vector_cube_mixed_amount_generated_batch[idx_in_mb] * vector_cube_mask_batch[idx_in_mb]).reshape(self.cube_shape + [self.num_channels_dict["discriminator"]]), amount_arr_scaled[idcMap[idx_in_total][0][0]:idcMap[idx_in_total][0][1], idcMap[idx_in_total][1][0]:idcMap[idx_in_total][1][1], idcMap[idx_in_total][2][0]:idcMap[idx_in_total][2][1], :] * counterArrInput01[idcMap[idx_in_total][0][0]:idcMap[idx_in_total][0][1], idcMap[idx_in_total][1][0]:idcMap[idx_in_total][1][1], idcMap[idx_in_total][2][0]:idcMap[idx_in_total][2][1], :]))
        overlappedArr = np.where(overlappedArr == 0., 1., overlappedArr)
        cluster_xyzt_pred = cluster_xyzt_pred / overlappedArr ## Overlapped imputations are averaged out.

        if False and self.debug: ## Check whether input observed mineral amount is preserved.
            assert(np.allclose(amount_arr_imputed * mask_arr_included, amount_arr * mask_arr_included, rtol = 1e-1, atol = 1e-4))
            print(f"The average difference in amount from reshape is {np.sum(np.abs(amount_arr_imputed * mask_arr_included - amount_arr * mask_arr_included)) / (np.sum(mask_arr_included) + self.small_delta)}")
            if filterWithDiscriminator or getPopulatedMask:
                assert(np.allclose(counter_arr_imputed * mask_arr_included, counter_arr_copied * mask_arr_included))
                print(f"The average difference in mask from reshape is {np.sum(np.abs(counter_arr_imputed * mask_arr_included - counter_arr_copied * mask_arr_included)) / (np.sum(mask_arr_included) + self.small_delta)}")
        
        ## I need to recover them again, because filter_num_observation_threshold_dict may make some observed entries skipped, so vector_cube_mixed_amount_generated_total may not contain all the observations.
        return cluster_xyzt_pred
    
    def clear_session(self):
        """Delete the Keras models to clean up the graph."""

        backend.clear_session()
        model_names = list(self.models_dict.keys())
        for model_name in model_names:
            del self.models_dict[model_name]
        
    def repeat_to_batch_size(self, loss, batch_size):
        """
            Parameters
            ----------
            loss: scalar, tensorflow float.
            batch_size: scalar, tensorflow int.

            Return
            ------
            loss copied batch_size times of shape (batch_size, ).
        """

        return tf.tile(tf.expand_dims(loss, axis= 0), [batch_size])

    def get_dummy_tensor_from_inputs(self, model_name, input_names, fill_value = 0.0):
        out = {}
        for input_name in input_names:
            out[input_name] = tf.fill(tf.shape(self.inputs_dict[model_name][input_name]), fill_value)
        return out

    def gen_loss_continuity_with_schatten_pnorm(self, vector_cube_generated_batch, vector_cube_mask_batch, vector_cube_amount_batch, pnorm = 0.5, pnorm_weight = 1.0):
        """ Get loss of continuity loss of generator. This loss encourage the imputed distribution becomse smooth. Generator tries to minimize this.

        Parameters
        ----------
        vector_cube_generated_batch : Tensor in shape (batch size, cube length)
            Imputed distribution by generator.
        
        vector_cube_mask_batch : Tensor in shape (batch size, cube length)
            Observabilities, 1 means observed 0 means unobserved.
        
        vector_cube_amount_batch : Tensor in shape (batch size, cube length)
            Input ground truth mineral amounts.
        
        Return
        ------
        : Tensor in shape Scalar or (batch size, ) 
        Calculated loss.
        """
        
        # vector_cube_mixed_amount_generated_batch = vector_cube_mask_batch * vector_cube_amount_batch + (1. - vector_cube_mask_batch) * vector_cube_generated_batch
        # vector_cube_estimated_mask_batch = self.discriminate(vector_cube_mixed_amount_generated_batch, vector_cube_hint_batch, shell_hint_amount_batch, shell_hint_counter_batch, is_training = True)
        
        # error_MSE = self.weight_MSE_fit * backend.mean(backend.pow((vector_cube_generated_batch - vector_cube_amount_batch) * vector_cube_mask_batch, 2)) / (backend.mean(vector_cube_mask_batch) + self.small_delta)
        # if self.verbose and self.kind_gen_loss_dis_entrybyentry == 'weighted_cross_entropy':
        #     print(f'dloss: {dloss}, MSE: {error_MSE}')

        # g_loss = - dloss + error_MSE

        ## Add difference between neighbors to encourage the continuity of imputation.
        cube_shape_with_channel = self.cube_shape + [self.num_channels_dict["generator"]]
        # mb_size = vector_cube_amount_batch.shape[0]
        # for i in range(1, 5):
        #     mb_size = mb_size // cube_shape_with_channel[i]
        reshaped_to_cube_mix_generated_amount = backend.reshape(vector_cube_generated_batch * (1 - vector_cube_mask_batch) + vector_cube_amount_batch * vector_cube_mask_batch, [tf.shape(vector_cube_generated_batch)[0]] + cube_shape_with_channel)
        sum_difference = 0.
        for i in range(self.cube_shape[0] - 1):
            sum_difference += backend.sum(backend.abs(reshaped_to_cube_mix_generated_amount[:, i, :, :, :] - reshaped_to_cube_mix_generated_amount[:, i + 1, :, :, :])) # consider batch dimension(shape[0])
        for j in range(self.cube_shape[1] - 1):
            sum_difference += backend.sum(backend.abs(reshaped_to_cube_mix_generated_amount[:, :, j, :, :] - reshaped_to_cube_mix_generated_amount[:, :, j + 1, :, :]))
        for k in range(self.cube_shape[2] - 1):
            sum_difference += backend.sum(backend.abs(reshaped_to_cube_mix_generated_amount[:, :, :, k, :] - reshaped_to_cube_mix_generated_amount[:, :, :, k + 1, :]))
        sum_difference = sum_difference * self.weight_continuity_loss_fit / (tf.cast(tf.shape(vector_cube_generated_batch)[0], tf.float32) * self.cube_shape[0] * self.cube_shape[1] * self.cube_shape[2] * self.num_channels_dict["generator"])
        # sum_difference = sum_difference * self.weight_continuity_loss_fit / backend.sum(backend.abs(reshaped_to_cube_mix_generated_amount))

        # def get_pnorm_of_one_batch(tensor_of_one_batch):
        #     schatten_pnorm_one_batch = 0.
        #     for channel_idx in range(self.num_channels):
        #         singular_values_vec = tf.linalg.svd(tensor_of_one_batch[:, :, :, channel_idx], compute_uv = False) + self.small_delta
        #         schatten_pnorm_one_batch = schatten_pnorm_one_batch + backend.pow(backend.mean(backend.pow(singular_values_vec, pnorm)), 1 / pnorm)
        #     return schatten_pnorm_one_batch
        # map_fn_result = backend.map_fn(get_pnorm_of_one_batch, reshaped_to_cube_mix_generated_amount)
        # schatten_pnorm_sum = backend.mean(map_fn_result) * pnorm_weight

        # for batch_idx in range(int((tf.shape(vector_cube_generated_batch)[0])):
        #     for channel_idx in range(self.num_channels):
        #         singular_values_vec = tf.linalg.svd(reshaped_to_cube_mix_generated_amount[batch_idx, :, :, :, channel_idx], compute_uv = False) + self.small_delta
        #         schatten_pnorm_sum = schatten_pnorm_sum + backend.pow(backend.mean(backend.pow(singular_values_vec, pnorm)), 1 / pnorm)

        # schatten_pnorm_sum = 0.
        # i = tf.constant(0)
        # def while_body(i, schatten_pnorm_sum):
        #     schatten_pnorm_one_batch = 0.
        #     for channel_idx in range(self.num_channels):
        #         singular_values_vec = tf.linalg.svd(reshaped_to_cube_mix_generated_amount[i, :, :, :, channel_idx], compute_uv = False) + self.small_delta
        #         schatten_pnorm_one_batch = schatten_pnorm_one_batch + backend.pow(backend.mean(backend.pow(singular_values_vec, pnorm)), 1 / pnorm)
        #     i = tf.add(i, 1)
        #     return [i, schatten_pnorm_sum + schatten_pnorm_one_batch]
        # def while_cond(i, schatten_pnorm_sum):
        #     return i < tf.shape(vector_cube_generated_batch)[0]
        # while_return = tf.while_loop(while_cond, while_body, [i, schatten_pnorm_sum])

        # return tf.tile(tf.expand_dims(sum_difference, axis= 0), [tf.shape(vector_cube_amount_batch)[0]]) # tries to minimize the differences between neighbor entries.
        return self.repeat_to_batch_size(sum_difference, tf.shape(vector_cube_amount_batch)[0]) # tries to minimize the differences between neighbor entries.
    
    def gen_loss_dis_entire(self, scalar_estimated_density):
        """Loss of generator in the fight with discriminator_entire. Generator tries to minimize this.

        Parameters
        ----------
        scalar_estimated_density : Tensor, (batch_size, )

        Return
        ------
        : Tensor in shape Scalar or (batch size, ) 
        Calculated loss.
        """

        return - (self.weight_of_dis_entire_density_for_gen_fit * scalar_estimated_density) # tries to increase scalar_estimated_density
        # loss input does not matter, just put any array with same shape

    def mse_loss(self, mask, true, pred, axes= 1):
        """Loss of generator for Mean Squared Error between input mineral amount and imputed mineral amount. Generator tries to minimize this.
        
        Parameters
        ----------
        vector_cube_generated_batch : Tensor in shape (batch size, length of cube)
            Imputed distribution by generator.
        
        vector_cube_amount_batch : Tensor in shape (batch size, cube length)
            Input ground truth mineral amounts.
        
        vector_cube_mask_batch : Tensor in shape (batch size, cube length)
            Observabilities, 1 means observed 0 means unobserved.
        
        Return
        ------
        : Tensor in shape Scalar or (batch size, ) 
        Calculated loss.
        """

        if not isinstance(axes, (tuple, list)): 
            axes = [axes]
        axes = None ## Don't need to leave batch dimension?

        # return tf.tile(tf.expand_dims(self.weight_MSE_fit * (backend.mean(backend.pow((vector_cube_generated_batch - vector_cube_amount_batch) * vector_cube_mask_batch, 2), axis = [1]) / (backend.mean(vector_cube_mask_batch, axis = [1]) + self.small_delta)), axis= 0), [tf.shape(vector_cube_amount_batch)[0]]) # tries to minimize the MSE
        return backend.mean(backend.pow((pred - true) * mask, 2), axis = axes) # tries to minimize the MSE

    def binary_prob_loss(self, true_prob, pred_prob, loss_mask = None, axes = 1):
        if not isinstance(axes, (tuple, list)): axes = [axes]
        if loss_mask is None:
            loss = - backend.mean(tf.reduce_mean(true_prob * backend.log(pred_prob + self.small_delta) + (1 - true_prob) * backend.log((1 - pred_prob) + self.small_delta), axis = axes))
        else:
            loss = - backend.mean(tf.reduce_mean(loss_mask * (true_prob * backend.log(pred_prob + self.small_delta) + (1 - true_prob) * backend.log((1 - pred_prob) + self.small_delta)), axis = axes))
        # return self.repeat_to_batch_size(loss, tf.shape(true_mask)[0])
        return loss

    def gen_prob_loss(self, pred_mask, true_mask= None, axes = 1):
        """Loss of generator in the fight with discriminator. Generator tries to minimize this.
        
        Parameters
        ----------
        vector_cube_mask_batch : Tensor in shape (batch size, cube length)
            Observabilities, 1 means observed 0 means unobserved.
        
        vector_cube_estimated_mask_batch : Tensor in shape (batch size, cube length)
            Estimated observabilities of certainties by discriminator.
        
        Return
        ------
        : Tensor in shape Scalar or (batch size, ) 
        Calculated loss.
        """

        if true_mask is None: true_mask = 0.0
        if not isinstance(axes, (tuple, list)): axes = [axes]

        if self.kind_gen_loss_dis_entrybyentry == 'weighted_cross_entropy':
            raise NotImplementedError()
            gloss = - backend.mean(backend.reduce_mean((1 - true_mask) * backend.log(pred_mask + self.small_delta), axis = axes) / (backend.reduce_mean(1 - true_mask, axis = axes) + self.small_delta))
        elif self.kind_gen_loss_dis_entrybyentry == 'cross_entropy':
            gloss = - backend.mean(tf.reduce_mean((1 - true_mask) * backend.log(pred_mask + self.small_delta), axis = axes))
        elif self.kind_gen_loss_dis_entrybyentry == 'mse':
            gloss = - backend.mean(backend.pow(true_mask - pred_mask, 2)) ## batch dimension does not matter thus there is no axis = [1]
        elif self.kind_gen_loss_dis_entrybyentry == 'matching_summation':
            gloss = - backend.mean((1 - true_mask) * pred_mask, axis = axes)
        else:
            raise Exception(f"Unsupported kind_gen_loss_dis_entrybyentry: {self.kind_gen_loss_dis_entrybyentry}")
        # return tf.tile(tf.expand_dims(gloss, axis= 0), [tf.shape(vector_cube_mask_batch)[0]]) # tries to maximize the real-like entries
        # return self.repeat_to_batch_size(gloss, tf.shape(true_mask)[0]) # tries to maximize the real-like entries
        return gloss

    def dis_loss_full(self, vector_cube_hint_mask_batch, vector_cube_mask_batch, vector_cube_estimated_mask_batch, kind = 0):
        """Loss of discriminator in the fight with generator. Discriminator tries to minimize this.
        
        Parameters
        ----------
        vector_cube_hint_mask_batch : Tensor in shape (batch size, cube length)
            Mask vector hinted with phint. Refer to GAIN paper for meaning of this.

        vector_cube_mask_batch : Tensor in shape (batch size, cube length)
            Observabilities, 1 means observed 0 means unobserved.
        
        vector_cube_estimated_mask_batch : Tensor in shape (batch size, cube length)
            Estimated observabilities of certainties by discriminator.
        
        Return
        ------
        : Tensor in shape Scalar or (batch size, ) 
        Calculated loss.
        """

        # test_arr = np.array([[1., 2.], [3., 4.], [5., 6.]])
        # print(np.mean(test_arr, axis = (1))) -> [1.5 3.5 5.5]
        # print(np.mean(test_arr)) -> 3.5

        d_loss = 0.
        if self.kind_gen_loss_dis_entrybyentry == 'weighted_cross_entropy':
            if kind == 0:
                # d_loss -= backend.mean((vector_cube_mask_batch - vector_cube_hint_mask_batch) * backend.log(vector_cube_estimated_mask_batch + self.small_delta)) / (backend.mean(vector_cube_mask_batch - vector_cube_hint_mask_batch) + self.small_delta) + backend.mean((1. - vector_cube_mask_batch) * backend.log(1. - vector_cube_estimated_mask_batch + self.small_delta)) / (backend.mean(1. - vector_cube_mask_batch) + self.small_delta)
                d_loss -= backend.mean(backend.mean(1. - vector_cube_mask_batch, axis = [1]) * backend.mean((vector_cube_mask_batch - vector_cube_hint_mask_batch) * backend.log(vector_cube_estimated_mask_batch + self.small_delta), axis = [1]) + backend.mean(vector_cube_mask_batch - vector_cube_hint_mask_batch, axis = [1]) * backend.mean((1. - vector_cube_mask_batch) * backend.log(1. - vector_cube_estimated_mask_batch + self.small_delta), axis = [1]))
            elif kind == 1:
                # d_loss -= backend.mean(vector_cube_mask_batch * backend.log(vector_cube_estimated_mask_batch + self.small_delta)) / (backend.mean(vector_cube_mask_batch) + self.small_delta) + backend.mean((1. - vector_cube_mask_batch) * backend.log(1. - vector_cube_estimated_mask_batch + self.small_delta)) / (backend.mean(1. - vector_cube_mask_batch) + self.small_delta)
                d_loss -= backend.mean(backend.mean(1. - vector_cube_mask_batch, axis = 1) * backend.mean(vector_cube_mask_batch * backend.log(vector_cube_estimated_mask_batch + self.small_delta), axis = [1]) + backend.mean(vector_cube_mask_batch, axis = [1]) * backend.mean((1. - vector_cube_mask_batch) * backend.log(1. - vector_cube_estimated_mask_batch + self.small_delta), axis = [1]))    
            else:
                raise Exception(f"Unsupported Discriminator Loss kind: {kind}")
        elif self.kind_gen_loss_dis_entrybyentry == 'mse':
            d_loss += backend.mean(backend.pow(vector_cube_mask_batch - vector_cube_estimated_mask_batch, 2)) ## batch dimension does not matter thus there is no axis = [1]
        elif self.kind_gen_loss_dis_entrybyentry == 'matching_summation':
            # d_loss -= backend.mean(vector_cube_mask_batch * backend.log(vector_cube_estimated_mask_batch + self.small_delta)) / (backend.mean(vector_cube_mask_batch) + self.small_delta) + backend.mean((1. - vector_cube_mask_batch) * backend.log(1. - vector_cube_estimated_mask_batch + self.small_delta)) / (backend.mean(1. - vector_cube_mask_batch) + self.small_delta)
            d_loss -= (backend.mean(1. - vector_cube_mask_batch, axis = [1]) * backend.mean(vector_cube_mask_batch * vector_cube_estimated_mask_batch, axis = [1]) + backend.mean(vector_cube_mask_batch, axis = [1]) * backend.mean((1. - vector_cube_mask_batch) * (1. - vector_cube_estimated_mask_batch), axis = [1]))
        else:
            raise Exception(f"Unsupported kind_gen_loss_dis_entrybyentry: {self.kind_gen_loss_dis_entrybyentry}")
        return self.repeat_to_batch_size(d_loss, tf.shape(vector_cube_mask_batch)[0])
    
    def dis_entire_loss(self, scalar_estimated_density, input_vector_cube_mask_batch):
        """Loss of discriminator_entire in the fight with generator. discriminator_entire tries to minimize this.
        
        Parameters
        ----------
        scalar_estimated_density : Tensor in shape (batch size, )
            The estimated proportion of 'real entries' / 'total entries'.

        input_vector_cube_mask_batch : Tensor in shape (batch size, cube length)
            Observabilities, 1 means observed 0 means unobserved.
                
        Return
        ------
        : Tensor in shape Scalar or (batch size, ) 
        Calculated loss.
        """

        if self.dis_entire_loss_type == "mse":
            return (scalar_estimated_density - backend.mean(input_vector_cube_mask_batch, axis = [1])) ** 2.
        else:
            raise Exception(f"Unsupported kind of loss: {self.dis_entire_loss_type}")

def pos_encode(self, x, y, z):
    """
    Return a positonally encoded vecoded vector from x,y,z parameters

    Parameters
    ----------
    x: The x position
    y: The y position
    z: The z position
    ----------
    """
    #sinusoidal encoding
    angles = tf.cast(1. / (10000 ** (tf.arange(0, self.num_channels_dict["generator"], 2).astype(tf.float32) / self.num_channels_dict["generator"])), dtype = tf.float32)

    #positions
    pos_x = np.arange(x).type(angles.type())
    pos_y = np.arange(y).type(angles.type())
    pos_z = np.arange(z).type(angles.type())

    #sin values of each position
    sin_x = tf.einsum("i,j->ij", pos_x, angles)
    sin_y = tf.einsum("i,j->ij", pos_y, angles)
    sin_z = tf.einsum("i,j->ij", pos_z, angles)

    #embeddings of x,y,z
    emb_x = tf.expand_dims(tf.expand_dims(tf.cat([sin_x.sin(), sin_x.cos()], axis=-1, name='concat'), 1), 1)
    emb_y = tf.expand_dims(tf.cat([sin_y.sin(), sin_y.cos()], axis=-1, name='concat') , 1)
    emb_z = tf.cat([sin_z.sin(), sin_z.cos()], axis=-1, name='concat')

    #transform to 4 dimensions
    emb_total = tf.zeros([x, y, z, self.num_channels_dict["generator"] * 3])
    emb_total[:,:,:,:self.num_channels_dict["generator"]] = emb_x
    emb_total[:,:,:,t: 2 * self.num_channels_dict["generator"]] = emb_y
    emb_total[:,:,:, 2 * self.num_channels_dict["generator"]:] = emb_z
    
    #return positionally encoded vector with batch size, number of channels
    return emb_total[None,:,:,:,:self.num_channels_dict["generator"]].repeat(self.batch_idx, 1, 1, 1, 1)