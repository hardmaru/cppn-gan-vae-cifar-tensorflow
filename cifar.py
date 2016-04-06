'''
loads all the data from cifar into memory.
some help from this insightful blog post: http://aidiary.hatenablog.com/entry/20151014/1444827123
assumes data is in cifar-10-batches-py, and cpkl'ed python version of data is downloaded/unzipped there.

0 - airplane
1 - automobile
2 - bird
3 - cat
4 - deer
5 - dog
6 - frog
7 - horse
8 - ship
9 - truck

'''
#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import random as random

class DataLoader():
  def __init__(self, batch_size=100, target_label=6, test_batch = False, all_images = True):
    self.data_dir = "./cifar-10-batches-py"
    self.batch_size = batch_size
    self.target_label = target_label

    datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    if test_batch == True:
      datafiles = ['test_batch']

    if all_images == True:
      datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

    def unpickle(f):
      fo = open(f, 'rb')
      d = cPickle.load(fo)
      fo.close()
      return d

    self.data = []

    for f in datafiles:
      d = unpickle(self.data_dir+'/'+f)
      data = d["data"]
      labels = np.array(d["labels"])
      nsamples = len(data)
      targets = np.where(labels == target_label)[0]
      for idx in targets:
        self.data.append(data[idx].reshape(3, 32, 32).transpose(1, 2, 0))

    self.data = np.array(self.data, dtype=np.float32)
    self.data /= 255.0

    self.num_examples = len(self.data)

    self.pointer = 0

    self.shuffle_data()

  def show_random_image(self):
    pos = 1
    for i in range(10):
      for j in range(10):
        plt.subplot(10, 10, pos)
        img = random.choice(self.data)
        # (channel, row, column) => (row, column, channel)
        plt.imshow(np.clip(img, 0.0, 1.0), interpolation='none')
        plt.axis('off')
        pos += 1
    plt.show()

  def show_image(self, image):
    '''
    image is in [height width depth]
    '''
    plt.subplot(1, 1, 1)
    plt.imshow(np.clip(image, 0.0, 1.0), interpolation='none')
    plt.axis('off')
    plt.show()

  def next_batch(self, batch_size):
    self.pointer += batch_size
    if self.pointer >= self.num_examples:
      self.pointer = 0
    result = []
    def random_flip(x):
      if np.random.rand(1)[0] > 0.5:
        return np.fliplr(x)
      return x
    for data in self.data[self.pointer:self.pointer+batch_size]:
      result.append(random_flip(data))
    return self.distort_batch(np.array(result, dtype=np.float32))

  def distort_batch(self, batch):
    batch_size = len(batch)
    row_distort = np.random.randint(0, 3, batch_size)
    col_distort = np.random.randint(0, 3, batch_size)
    result = np.zeros(shape=(batch_size, 30, 30, 3), dtype=np.float32)
    for i in range(batch_size):
      result[i, :, :, :] = batch[i, row_distort[i]:row_distort[i]+30, col_distort[i]:col_distort[i]+30, :]
    return result

  def shuffle_data(self):
    self.data = np.random.permutation(self.data)


