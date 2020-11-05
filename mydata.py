import logging
import numpy as np
from glob import glob
import cv2
import os

RANDOM_SEED = 42
RNG = np.random.RandomState(42)

logger = logging.getLogger(__name__)
input_size = 256

class MyData(object):
    def __init__(self, type, batch_size, input_size=input_size):
        self.data_dir = 'data/mydata_for_%s' % type
        self.img_files = glob(os.path.join(self.data_dir, '*[jpg,png,JPG,PNG]'))
        self.num_img = len(self.img_files)
        self.batch_size = batch_size
        self.num_batches = np.ceil(self.num_img / self.batch_size).astype(int)
        self.input_size = input_size
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        np.random.shuffle(self.img_files)
        batch_images = np.zeros((self.batch_size, self.input_size, self.input_size, 3), dtype=np.float32)
        num = 0
        if self.batch_count < self.num_batches:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_img:
                    index -= self.num_img
                img_path = self.img_files[index]
                img_arr = cv2.resize(cv2.imread(img_path),(self.input_size, self.input_size))
                if img_arr.ndim < 3:
                    img_arr = np.expand_dims(img_arr,axis=-1)
                img_float = img_arr.astype(np.float32)/255*2-1
                batch_images[num, :, :, :] = img_float
                num += 1
            self.batch_count += 1
            return batch_images
        else:
            self.batch_count = 0
            np.random.shuffle(self.img_files)
            raise StopIteration


if __name__ == "__main__":
    data = MyData(type='test', batch_size=3)
    for i in range(1):
        for t,d in enumerate(data):
            print(d)
            cv2.imshow('0', d[0])
            cv2.imshow('1', d[1])
            cv2.imshow('2', d[2])
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(data.num_batches)
