import numpy as np
from PIL import Image
import scipy

import caffe

np.set_printoptions(threshold='nan')

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('/home/chenliu/Projects/DeepLearningToyExamples/caffe/data/cifar10/images/image_99.png')
im = Image.open('/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Data/competitor_99.png')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
#in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))
#in_[:, :, :] = 0

# load net
net = caffe.Net('/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Models/cifar_train.prototxt', '/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Models/_iter_1000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
#net.blobs['data'].reshape(1, *in_.shape)
#net.blobs['data'].reshape(1, 3, 32, 32)
net.reshape()
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
data = net.blobs['data'].data
label = net.blobs['label'].data
data = np.squeeze(data[0, :, :, :])
scipy.misc.imsave('test.png', data);

pred = net.blobs['ip2'].data
prob = net.blobs['accuracy'].data
#prob = np.squeeze(prob[0,:,:,:])
#label = np.argmax(prob, axis=0);

print "label: "
print pred
#print prob
#print label
#print label
