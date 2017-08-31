import tensorflow as tf
import numpy as np

import VAE

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def transform_with_vae(vae_load_path, network_architecture, transform_save_path):
    vae = VAE.VariationalAutoencoder(network_architecture)
    vae.load(filepath=vae_load_path)

    mnist = read_data_sets('MNIST_data', one_hot=True)
    images = mnist.train.images

    coded_images = vae.encode(images)
    np.save(file=transform_save_path, arr=coded_images)
    return 0

def save_distance_matrix(coordinates_file, output_file):
    
