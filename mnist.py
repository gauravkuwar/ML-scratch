import numpy as np
import gzip
import matplotlib.pyplot as plt

class MNIST:
    def __init__(self, num_images_train=60000, num_images_test=10000):
        self.num_images_train = num_images_train
        self.num_images_test = num_images_test
        self.num_of_rows = 28
        self.num_of_cols = 28
        self.train, self.test = [(0, 0)], [(0, 0)]
    
    def load(self, t='train'):        
        if t == 'train':
            num_images = self.num_images_train
            fn_img = 'mnist/train-images-idx3-ubyte.gz'
            fn_labels = 'mnist/train-labels-idx1-ubyte.gz'
        else:
            num_images = self.num_images_test
            fn_img = 'mnist/t10k-images-idx3-ubyte.gz'
            fn_labels = 'mnist/t10k-labels-idx1-ubyte.gz'
        
        with gzip.open(fn_labels,'r') as fl:
            with gzip.open(fn_img, 'r') as fi:
                buf = fi.read(self.num_of_rows * self.num_of_cols * num_images + 16)
                data = np.frombuffer(buf, dtype=np.uint8, offset=16).astype(np.float32) \
                         .reshape(num_images, 1, self.num_of_rows, self.num_of_cols) \
                         / 255
                
                buf = fl.read(num_images + 8)
                labels = np.frombuffer(buf, dtype=np.uint8, offset=8).astype(np.int32).reshape(num_images)

        if t == 'train':
            self.train_data = data
            self.train_labels = labels
        else:
            self.test_data = data
            self.test_labels = labels
                            
    def show(self, idx, t='train'):
        if t == 'train':
            print(self.train[idx][1])
            plt.imshow(np.asarray(self.train[idx][0]).squeeze())
        else:
            print(self.test[idx][1])
            plt.imshow(np.asarray(self.test[idx][0]).squeeze())
            
        plt.show()