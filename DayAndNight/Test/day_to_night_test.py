import numpy as np
from PIL import Image
from PIL import ImageEnhance
import scipy
import sys

import caffe

np.set_printoptions(threshold='nan')

day_image = Image.open('/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Data/day_2.jpg')

image_size = day_image.size
scale = 4
day_to_night_scaling = 0.15
output_image = Image.new('L', (image_size[0] / scale, image_size[1] / scale))
output_pixels = output_image.load()

result_image = day_image.resize((image_size[0] / scale, image_size[1] / scale))
#enhancer = ImageEnhance.Brightness(result_image)
#enhancer.enhance(0.15)
#result_image.show()
prob = [line.rstrip('\n') for line in open('prob.txt')]
prob = [line.split(' ') for line in prob]

#prob = [line[0] + ' ' + line[1] + ' ' + line[2] + '\n' for line in prob]
#with open('test.txt', 'wb') as f:
  #f.writelines(prob)
#exit()

lab_image = result_image.convert('LAB')
result_pixels = lab_image.load()
for y in range(result_image.size[1]):
  for x in range(result_image.size[0]):
    pixel = result_pixels[x, y]
    #pixel = (int(pixel[0] * day_to_night_scaling), int(pixel[1] * day_to_night_scaling), int(pixel[2] * day_to_night_scaling))
    pixel[0] *= day_to_night_scaling
    result_pixels[x, y] = pixel
    
for line in prob:
  color = result_pixels[int(line[0]) / scale, int(line[1]) / scale]
  if float(line[2]) < 0.3:
    continue
  #color = (int(min(color[0] + float(line[2]) * 255, 255)), int(min(color[1] + float(line[2]) * 255, 255)), int(min(color[2] + float(line[2]) * 128, 255)))
  color[0] = int(float(line[2]) * 255)
  result_pixels[int(line[0]) / scale, int(line[1]) / scale] = color
result_image = lab_image.convert('RGB')
result_image.save('result_image.png')
exit()

# load net
net = caffe.Net('/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Models/differentiator_deploy.prototxt', '/home/chenliu/Projects/DeepLearningToyExamples/DayAndNight/Models/differentiator_iter_1000.caffemodel', caffe.TEST)
output_file = open('test.txt', 'wb')
skip = 0
for y in range(image_size[1]):
  for x in range(image_size[0]):
    if y % scale != 0 or x % scale != 0:
      continue
    if skip > 0:
      skip -= 1
      continue
 
    im = day_image.crop((x, y, x + 112, y + 112))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ = in_.transpose((2,0,1))

    net.blobs['data'].data[...] = in_

    data = net.blobs['data'].data
    data = np.squeeze(data[0, :, :, :])
    data = data.transpose((1,2,0))
    net.forward()

    prob = net.blobs['prob'].data
    prob = np.squeeze(prob[0,:])
    label = np.argmax(prob, axis=0);
    output_pixels[x / scale, y / scale] = int(prob[0] * 255)

    if prob[0] < 0.3:
      skip = 4
    print x, y, prob[0]
    
#print label.shape
#print prob
    
