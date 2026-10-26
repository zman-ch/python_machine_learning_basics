import struct
from pathlib import Path

import numpy as np


def load_fashion_partition(data_root, *, training):
    data_directory = Path(data_root) / 'FashionMNIST' / 'raw'
    prefix = 'train' if training else 't10k'
    image_bytes = (data_directory / f'{prefix}-images-idx3-ubyte').read_bytes()
    label_bytes = (data_directory / f'{prefix}-labels-idx1-ubyte').read_bytes()
    if len(image_bytes) < 16 or len(label_bytes) < 8:
        raise ValueError('Truncated Fashion-MNIST IDX header.')
    image_magic, image_count, rows, columns = struct.unpack('>IIII', image_bytes[:16])
    label_magic, label_count = struct.unpack('>II', label_bytes[:8])
    if (image_magic, label_magic, rows, columns) != (2051, 2049, 28, 28):
        raise ValueError('Invalid Fashion-MNIST IDX format or image dimensions.')
    if image_count != label_count or image_count == 0:
        raise ValueError('Fashion-MNIST image and label counts do not match.')
    if len(image_bytes) != 16 + image_count * rows * columns or len(label_bytes) != 8 + label_count:
        raise ValueError('Fashion-MNIST IDX payload length does not match its header.')
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(image_count, rows, columns)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    if np.any(labels > 9):
        raise ValueError('Fashion-MNIST labels must be integers from 0 to 9.')
    return images, labels