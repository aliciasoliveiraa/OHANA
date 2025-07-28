import numpy as np
from   scipy.ndimage import rotate, shift
import cv2
from   skimage.metrics import structural_similarity
import tensorflow as tf


def transpose(img, r, tx, ty, channel_last=True):
    if img.ndim == 2:
        s = shift(rotate(img, r, [0, 1], order=1, reshape=False), [tx, ty], order=1)
    elif img.ndim == 3:
        if channel_last:
            s = shift(rotate(img, r, [0, 1], order=1, reshape=False), [tx, ty, 0], order=1)
        else:
            s = shift(rotate(img, r, [1, 2], order=1, reshape=False), [0, tx, ty], order=1)
    elif img.ndim == 4:
        s = shift(rotate(img, r, [1, 2], order=1, reshape=False), [0, tx, ty, 0], order=1)
    else:
        raise TypeError

    return s


def rand_range(range):
    return (np.random.rand()-0.5)*2*range


def generate_rand_par(num):
    x = (np.random.rand(num)-.5)*num/(num-1)
    return x


def rot_tra_argumentation(y, x, num_contrast):
    size = y[0].shape[0]

    x_copy = x
    y_copy = y
    for i in range(size):
        r = rand_range(10)
        tx = rand_range(5)
        ty = rand_range(5)
        for j in range(num_contrast):
            y_copy[j][i,:,:,:] = transpose(y[j][i,:,:,:], r, tx, ty)
            x_copy[j][i,:,:,:] = transpose(x[j][i,:,:,:], r, tx, ty)

    return y_copy, x_copy


def save_image(file_name, image):
    image = np.squeeze(image)*255
    cv2.imwrite(file_name, image)


def merge_images(x, y, p):
    size = x[0].shape[0]
    h = x[0].shape[1]
    w = x[0].shape[2]
    out = np.zeros([size, h*3, w*3])
    for i in range(size):
        out[i, 0:h, 0:w] = x[0][i,:,:,0]
        out[i, 0:h, w:w*2] = x[1][i,:,:,0]
        out[i, 0:h, w*2:w*3] = x[2][i,:,:,0]
        out[i, h:h*2, 0:w] = y[0][i,:,:,0]
        out[i, h:h*2, w:w*2] = y[1][i,:,:,0]
        out[i, h:h*2, w*2:w*3] = y[2][i,:,:,0]
        out[i, h*2:h*3, 0:w] = p[0][i,:,:,0]
        out[i, h*2:h*3, w:w*2] = p[1][i,:,:,0]
        out[i, h*2:h*3, w*2:w*3] = p[2][i,:,:,0]
    return out


def ssim(img, ref):
    return structural_similarity(img, ref, data_range=1.)


def test_ssim(x_img, y_img, p_img):
    x_ssim = []
    p_ssim = []
    for i in range(x_img.shape[0]):
        x_ssim.append(ssim(x_img[i,:,:,0], y_img[i,:,:,0]))
        p_ssim.append(ssim(p_img[i,:,:,0], y_img[i,:,:,0]))
    return np.mean(x_ssim), np.mean(p_ssim)


def nmi(img, ref, bins=16):
    eps = 1e-10
    hist = np.histogram2d(img.flatten(), ref.flatten(), bins=bins)[0]
    pxy = hist / np.sum(hist)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2((pxy[nzs] + eps) / (px_py[nzs] + eps)))
    entx = - np.sum(px * np.log2(px + eps))
    enty = - np.sum(py * np.log2(py + eps))

    return (2 * mi + eps) / (entx + enty + eps)


def test_nmi(x_img, y_img, p_img):
    x_nmi = []
    p_nmi = []
    for i in range(x_img.shape[0]):
        x_nmi.append(nmi(x_img[i,:,:,0], y_img[i,:,:,0]))
        p_nmi.append(nmi(p_img[i,:,:,0], y_img[i,:,:,0]))
    return np.mean(x_nmi), np.mean(p_nmi)


def nrmse(img, ref):
    rmse = np.sqrt(np.mean((img-ref)**2))
    return rmse/np.mean(ref)


def test_nrmse(x_img, y_img, p_img):
    x_nrmse = []
    p_nrmse = []
    for i in range(x_img.shape[0]):
        x_nrmse.append(nrmse(x_img[i,:,:,0], y_img[i,:,:,0]))
        p_nrmse.append(nrmse(p_img[i,:,:,0], y_img[i,:,:,0]))
    return np.mean(x_nrmse), np.mean(p_nrmse)


#Function validate_axis from tf_utils.py

def validate_axis(axis, input_shape):
  """Validate an axis value and returns its standardized form.

  Args:
    axis: Value to validate. Can be an integer or a list/tuple of integers.
      Integers may be negative.
    input_shape: Reference input shape that the axis/axes refer to.

  Returns:
    Normalized form of `axis`, i.e. a list with all-positive values.
  """
  input_shape = tf.TensorShape(input_shape)
  rank = input_shape.rank
  if not rank:
    raise ValueError(
        f'Input has undefined rank. Received: input_shape={input_shape}')

  # Convert axis to list and resolve negatives
  if isinstance(axis, int):
    axis = [axis]
  else:
    axis = list(axis)
  for idx, x in enumerate(axis):
    if x < 0:
      axis[idx] = rank + x

  # Validate axes
  for x in axis:
    if x < 0 or x >= rank:
      raise ValueError(
          'Invalid value for `axis` argument. '
          'Expected 0 <= axis < inputs.rank (with '
          f'inputs.rank={rank}). Received: axis={tuple(axis)}')
  if len(axis) != len(set(axis)):
    raise ValueError(f'Duplicate axis: {tuple(axis)}')
  return axis

#Functions flat_layer_weights_dict and unflat_layer_weights_dict from NVFlare/nvflare/app_opt/tf/utils.py

SPECIAL_KEY = "_nvf_"


def flat_layer_weights_dict(data: dict):
    """Flattens layer weights dict."""
    result = {}
    for layer_name, weights in data.items():
        if len(weights) != 0:
            # If the original layer get_weights return: {"layer0": [array1, array2]}
            # We will convert it to: {"layer0_nvf_0": array1, "layer0_nvf_1": array2}
            for i, item in enumerate(weights):
                result[f"{layer_name}{SPECIAL_KEY}{i}"] = item
    return result


def unflat_layer_weights_dict(data: dict):
    """Unflattens layer weights dict."""
    result = {}
    for k, v in data.items():
        if SPECIAL_KEY in k:
            # If the weight is: {"layer0_nvf_0": array1, "layer0_nvf_1": array2}
            # We will convert it back to: {"layer0": [array1, array2]} and load it back
            layer_name, _ = k.split(SPECIAL_KEY)
            if layer_name not in result:
                result[layer_name] = []
            result[layer_name].append(v)
    return result
