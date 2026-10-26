from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import numpy as np


MNIST_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
MNIST_SHA256 = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'


def load_mnist(cache_directory=None):
    if cache_directory is None:
        cache_directory = Path.home() / '.cache' / 'zhaw-ml-bi'
    archive = Path(cache_directory) / 'mnist.npz'
    if archive.exists():
        payload = archive.read_bytes()
    else:
        with urlopen(MNIST_URL, timeout=60) as response:
            payload = response.read()
    if sha256(payload).hexdigest() != MNIST_SHA256:
        raise ValueError(f'MNIST checksum mismatch for {archive}; remove a corrupt cached file and retry.')
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(payload)
    with np.load(archive, allow_pickle=False) as data:
        training_images, training_labels = data['x_train'], data['y_train']
        test_images, test_labels = data['x_test'], data['y_test']
    return (training_images, training_labels), (test_images, test_labels)