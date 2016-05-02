import numpy as np
from PIL import Image
import scipy
import sys

import caffe

np.set_printoptions(threshold='nan')

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('/home/chenliu/Projects/DeepLearningToyExamples/caffe/data/cifar10/images/image_99.png')
im = Image.open('/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Data/fakenight/400.jpg')
im = im.resize((227, 227), Image.ANTIALIAS)
im.save("resized.jpg")
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
#in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))
#scipy.misc.imsave('test_input.png', in_);
#in_[:, :, :] = 0

# load net
net = caffe.Net('/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Models/labeling_deploy.prototxt', '/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Models/labeling_iter_3000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
#net.blobs['data'].reshape(1, *in_.shape)
#net.blobs['data'].reshape(1, 3, 32, 32)
#net.reshape()
net.blobs['data'].data[...] = in_

data = net.blobs['data'].data
data = np.squeeze(data[0, :, :, :])
data = data.transpose((1,2,0))
print data.shape
#print data.dtype
#sys.exit("quit")
test_image = Image.fromarray(np.int32(data), 'RGB')
test_image.save('test.jpg')

# run net and take argmax for prediction
net.forward()
#scipy.misc.toimage(data, cmin=0.0, cmax=...).save('test_input.png')
#scipy.misc.imsave('test.png', data);

prob = net.blobs['prob'].data
prob = np.squeeze(prob[0,:])
label = np.argmax(prob, axis=0);

print "label: "
print prob
print label
#print prob
#print label
#print label
