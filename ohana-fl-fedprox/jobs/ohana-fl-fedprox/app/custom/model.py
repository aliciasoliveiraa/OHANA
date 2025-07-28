import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.saving import register_keras_serializable
tf.random.set_seed(22)
np.random.seed(22)


"""Group normalization layer"""

from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import InputSpec
from keras.layers import Layer
from utils import validate_axis

class GroupNormalization(Layer):
    def __init__(
        self,
        groups=1,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        validate_axis(self.axis, input_shape)

        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                f"Axis {self.axis} of input tensor should have a defined "
                "dimension but the layer received an input with shape "
                f"{input_shape}."
            )

        if self.groups == -1:
            self.groups = dim

        if dim < self.groups:
            raise ValueError(
                f"Number of groups ({self.groups}) cannot be more than the "
                f"number of channels ({dim})."
            )

        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must be a multiple "
                f"of the number of channels ({dim})."
            )

        self.input_spec = InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs, mask=None):
        input_shape = tf.shape(inputs)

        if mask is None:
            mask = tf.ones_like(inputs)
        else:
            mask = tf.broadcast_to(mask, input_shape)

        reshaped_inputs = self._reshape_into_groups(inputs)
        reshaped_mask = self._reshape_into_groups(mask)

        normalized_inputs = self._apply_normalization(
            reshaped_inputs=reshaped_inputs,
            input_shape=input_shape,
            reshaped_mask=reshaped_mask,
        )

        return tf.reshape(normalized_inputs, input_shape)

    def _reshape_into_groups(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = [input_shape[i] for i in range(inputs.shape.rank)]

        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs

    def _apply_normalization(
        self,
        *,
        reshaped_inputs,
        reshaped_mask,
        input_shape,
    ):
        group_reduction_axes = list(range(1, reshaped_inputs.shape.rank))

        axis = self.axis - 1
        group_reduction_axes.pop(axis)

        mask_weights = tf.cast(reshaped_mask, reshaped_inputs.dtype)

        mean, variance = tf.nn.weighted_moments(
            reshaped_inputs,
            axes=group_reduction_axes,
            frequency_weights=mask_weights,
            keepdims=True,
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * backend.int_shape(input_shape)[0]

        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)

        return broadcast_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}
    


""" Motion correction network """


class Encoder(keras.Model):
    def __init__(self, initial_filters=64):
        super(Encoder, self).__init__()

        self.filters = initial_filters

        self.conv1 = keras.layers.Conv2D(self.filters, kernel_size=7, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(self.filters*2, kernel_size=3, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(self.filters*4, kernel_size=3, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.n1 = GroupNormalization()
        self.n2 = GroupNormalization()
        self.n3 = GroupNormalization()


    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.n1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.n2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.n3(x, training=training)
        x = tf.nn.relu(x)

        return x


class Residual(keras.Model):
    def __init__(self, initial_filters=256):
        super(Residual, self).__init__()

        self.filters = initial_filters

        self.conv1 = keras.layers.Conv2D(self.filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(self.filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.in1 = GroupNormalization()
        self.in2 = GroupNormalization()

    def call(self, x, training=True):
        inputs = x

        x = self.conv1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = tf.nn.relu(x)

        x = tf.add(x, inputs)

        return x


class Decoder(keras.Model):
    def __init__(self, initial_filters=128):
        super(Decoder, self).__init__()

        self.filters = initial_filters

        self.conv1 = keras.layers.Conv2DTranspose(self.filters, kernel_size=3, strides=2, padding='same',
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2DTranspose(self.filters//2, kernel_size=3, strides=2, padding='same',
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(1, kernel_size=7, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.in1 = GroupNormalization()
        self.in2 = GroupNormalization()
        self.in3 = GroupNormalization()

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.in1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.in2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.in3(x, training=training)
        x = tf.nn.relu(x)

        return x


class MC_Net(keras.Model):
    def __init__(self,
                 img_size=256,
                 num_filter=32,
                 num_contrast=4,
                 num_res_block=9):
        super(MC_Net, self).__init__()

        self.img_size = img_size
        self.filters = num_filter
        self.num_contrast = num_contrast
        self.num_res_block = num_res_block

        self.encoder_list = []
        for _ in range(num_contrast):
            self.encoder_list.append(Encoder(initial_filters=self.filters))

        self.res_block_list = []
        for _ in range(num_res_block):
            self.res_block_list.append(Residual(initial_filters=self.filters*4*num_contrast))

        self.decoder_list = []
        for _ in range(num_contrast):
            self.decoder_list.append(Decoder(initial_filters=self.filters*2))

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(MC_Net, self).build(input_shape)

    def call(self, x, training=True):
        x_list = []
        for i in range(self.num_contrast):
            x_list.append(self.encoder_list[i](x[i], training=training))
        x = tf.concat(x_list, axis=-1)

        for i in range(self.num_res_block):
            x = (self.res_block_list[i](x, training=training))

        y = tf.split(x, num_or_size_splits=self.num_contrast, axis=-1)
        
        y_list = []
        for i in range(self.num_contrast):
            y_list.append(self.decoder_list[i](y[i], training=training))
                
        return y_list


def ssim_loss(img1, img2):
    ssim = -tf.math.log((tf.image.ssim(img1, img2, max_val=1.0)+1)/2)
    return ssim


def vgg_layers(layer_names):
    local_weights_path = '/path/to/moana-fl-fedprox/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights=local_weights_path, input_shape=(256, 256, 3))
    
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def vgg_loss(img1, img2, loss_model):
    img1 = tf.repeat(img1, 3, -1)
    img2 = tf.repeat(img2, 3, -1)
    
    mean = tf.reduce_mean(tf.square(loss_model(img1) - loss_model(img2)))
    
    return mean


@register_keras_serializable()
def make_custom_loss(l1, l2, loss_model):
    def custom_loss(y_true, y_pred):
        return l1*ssim_loss(y_true, y_pred) + l2*vgg_loss(y_true, y_pred, loss_model)

    return custom_loss
