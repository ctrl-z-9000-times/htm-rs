""" An MNIST classifier using the Spatial Pooler. """

from pathlib import Path
import gzip
import htm_rs
import numpy as np
import random
from htm.algorithms import Classifier
import htm

def load_mnist():
    """See: http://yann.lecun.com/exdb/mnist/ for MNIST download and binary file format spec."""
    def int32(b):
        i = 0
        for char in b:
            i *= 256
            # i += ord(char)    # python2
            i += char
        return i

    def load_labels(file_name):
        file_name = Path(file_name).resolve()
        with gzip.open(file_name, 'rb') as f:
            raw = f.read()
            assert(int32(raw[0:4]) == 2049)  # Magic number
            labels = []
            for char in raw[8:]:
                # labels.append(ord(char))      # python2
                labels.append(char)
        return labels

    def load_images(file_name):
        file_name = Path(file_name).resolve()
        with gzip.open(file_name, 'rb') as f:
            raw = f.read()
            assert(int32(raw[0:4]) == 2051)    # Magic number
            num_imgs   = int32(raw[4:8])
            rows       = int32(raw[8:12])
            cols       = int32(raw[12:16])
            assert(rows == 28)
            assert(cols == 28)
            img_size   = rows*cols
            data_start = 4*4
            imgs = []
            for img_index in range(num_imgs):
                vec = raw[data_start + img_index*img_size : data_start + (img_index+1)*img_size]
                # vec = [ord(c) for c in vec]   # python2
                vec = list(vec)
                vec = np.array(vec, dtype=np.uint8)
                buf = np.reshape(vec, (rows, cols))
                imgs.append(buf)
            assert(len(raw) == data_start + img_size * num_imgs)   # All data should be used.
        return imgs

    train_labels = load_labels('./MNIST_data/train-labels-idx1-ubyte.gz')
    train_images = load_images('./MNIST_data/train-images-idx3-ubyte.gz')
    test_labels  = load_labels('./MNIST_data/t10k-labels-idx1-ubyte.gz')
    test_images  = load_images('./MNIST_data/t10k-images-idx3-ubyte.gz')

    return train_labels, train_images, test_labels, test_images


def test_mnist():

    train_labels, train_images, test_labels, test_images = load_mnist()

    training_data = list(zip(train_images, train_labels))
    test_data     = list(zip(test_images, test_labels))
    random.shuffle(training_data)

    # DEBUGGING!
    # training_data = training_data[:10000]
    # test_data = test_data[:500]

    # Setup the AI.
    sp = htm_rs.SpatialPooler(
            num_cells  = 1400,
            num_active =  140,
            num_steps = 0,
            threshold = .1,
            potential_pct = .2,
            learning_period = 100,
            max_num_patterns = 20,
            weight_gain = 100,
            boosting_period = 1400,
            seed=None,)

    stats = htm_rs.Stats(1e6)
    sdrc = Classifier()

    # Training Loop
    for img, lbl in training_data:
        img = [bool(x) for x in img.reshape(-1) > 100]
        enc = htm_rs.SDR.from_dense(img)
        rs_columns = sp.advance( enc, True )
        cpp_columns = htm.SDR([rs_columns.num_cells()])
        cpp_columns.sparse = rs_columns.sparse()
        sdrc.learn( cpp_columns, lbl )
        stats.update(rs_columns)
        print('.', end='', flush=1)
    print()
    print(sp)
    print(stats)

    # Testing Loop
    score = 0
    for img, lbl in test_data:
        img = [bool(x) for x in img.reshape(-1) > 100]
        enc = htm_rs.SDR.from_dense(img)
        rs_columns = sp.advance( enc, False )
        cpp_columns = htm.SDR([rs_columns.num_cells()])
        cpp_columns.sparse = rs_columns.sparse()
        if lbl == np.argmax( sdrc.infer( cpp_columns ) ):
            score += 1
    score = score / len(test_data)

    print('Score:', 100 * score, '%')
    assert score > .95


if __name__ == '__main__':
    test_mnist()
