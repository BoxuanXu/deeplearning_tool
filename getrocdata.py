import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple

import mxnet as mx

from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def get_label():
   with open('val.lst',"r") as f:
       lines = f.readlines()
       res = []
       for line in lines:
           lineData=line.strip().split(' ')
           res.append(lineData)
   return res

Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint('/GIS/data/model/resnet101', 30)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,128,128))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

def get_image(path, show=False):
    # download and show the image
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    #if show:
    #     plt.imshow(img)
    #     plt.axis('off')
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
    #print(prob)
    return prob
    # print the top-5
    #prob = np.squeeze(prob)
    #a = np.argsort(prob)[::-1]
    #for i in a[0:5]:
    #    print((prob[i]))

if __name__ == '__main__':
  res = get_label()
  labels = np.empty((len(res), 5))
  probs = np.empty((len(res), 5))
  i = 0
  for line in res:
      label = np.array([0,0,0,0,0])
      label[int(line[0])] = 1
      labels[i] = label
      #print(label)
      prob = predict("val/" + line[1])
      probs[i] = prob
      i = i + 1
  print(labels)
  print(probs)
  n_classes = 5 
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), probs.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
  # Finally average it and compute AUC
  mean_tpr /= n_classes
  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  lw=2
  plt.figure()
  plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

  plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

  colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'Magenta', 'goldenrod'])
  for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Some extension of Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.savefig('roc101.jpg')

