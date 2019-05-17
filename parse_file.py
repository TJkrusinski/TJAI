import gzip
import numpy as np

files = {
    'training_images': './MNIST/train-images-idx3-ubyte.gz',
    'training_labels': './MNIST/train-labels-idx1-ubyte.gz',
    'test_images': './MNIST/t10k-images-idx3-ubyte.gz',
    'test_labels': './MNIST/t10k-labels-idx1-ubyte.gz'
}

class DataFile: 
    def __init__(self, file):
        self.file_name = files[file]
        self.file = None
        self.magic = 0
        self.items = 0
        self.image_length = 28 * 28
        self.label_length = 1
        self.is_lables = False
        self.is_images = False
        self.load()

    def load(self):
        self.file = gzip.open(self.file_name, 'rb')
        self.file.seek(0, 0)
        self.magic = int.from_bytes(self.file.read(4), byteorder='big', signed=False)
        self.file.seek(4, 0)
        self.items = int.from_bytes(self.file.read(4), byteorder='big', signed=False)
        self.file.seek(0, 0)

    def __len__(self):
        return self.items

class Image:
    def __init__(self, pixels, label=None):
        self.pixels = pixels
        self.label = None
    
    def __getitem__(self, index):
        return self.pixels[index]
    
    def __len__(self):
        return len(self.pixels)

    def print(self):
        number = ""
        for i in range(0, 28):
            for j in range(0, 28):
                pixel = self.pixels[i * 28 + j] * 255.0
                if pixel < 50:
                    number += "  "
                elif pixel < 100:
                    number += ".."
                elif pixel < 150:
                    number += "::"
                elif pixel < 200:
                    number += "##"
                else:
                    number += "@@"

            number += "\n\r"
        return number
        


class ImageData(DataFile):

    def seek(self, pos):
        pre = 4 + 4 + 4 + 4 + (pos * self.image_length)
        self.file.seek(pre, 0)

    def __getitem__(self, index):
        pre = 4 + 4 + 4 + 4 + (index * self.image_length)
        pixels = np.empty(28*28, dtype=np.uint8)
        for i in range(0, self.image_length):
            self.file.seek(pre + i, 0)
            pixels[i] = int.from_bytes(self.file.read(1), byteorder='big', signed=False)
        return pixels

class LabelData(DataFile):

    def seek(self, pos):
        pre = 4 + 4 + (pos * self.label_length)
        self.file.seek(pre, 0)

    def __getitem__(self, index):
        self.seek(index)
        number = int.from_bytes(self.file.read(1), byteorder='big', signed=False)
        return number
        #label[number] = 1
        #return label