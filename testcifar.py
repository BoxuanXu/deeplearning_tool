import matplotlib.pyplot as plt
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple

import mxnet as mx
Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint('/GIS/model/cifar', 15)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

def get_image(path, show=False):
    # download and show the image
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (128, 128))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img
 
def predict(path):
    img = get_image(path, show=True)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print((prob[i]))

predict('/home/test_images/cat.jpg')
