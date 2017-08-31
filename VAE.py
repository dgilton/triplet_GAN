import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import pickle

np.random.seed(0)
tf.set_random_seed(0)

mnist = read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples


# Xavier Initialization
def xavier_init(fan_in, fan_out, constant=1):
    """Initialize network weights with Xavier Initialization"""
    low = -constant*np.sqrt(6/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class VariationalAutoencoder(object):
    """Variational Autoencoder with bog-standard settings. Stolen from jmetzen"""

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.vae_input = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        self._create_network()

        self._create_loss_optimizer()

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

    def _create_network(self):
        """Create the network"""
        # Initialize weights
        self.network_weights = self._initialize_weights(**self.network_architecture)

        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["encoder_weights"], self.network_weights["encoder_biases"])

        # Draw sample from Gaussian Distribution
        code_size = self.network_architecture["code_size"]
        eps = tf.random_normal((self.batch_size, code_size), 0, 1, dtype=tf.float32)

        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.x_decode_mean = \
            self._decoding_network(self.network_weights["decoder_weights"], self.network_weights["decoder_biases"])



    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2, n_input, code_size):
        all_weights = dict()
        all_weights['encoder_weights'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, code_size)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, code_size))}
        all_weights['encoder_biases'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([code_size], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([code_size], dtype=tf.float32))}
        all_weights['decoder_weights'] = {
            'h1': tf.Variable(xavier_init(code_size, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['decoder_biases'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        """Create encoding network"""
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.vae_input, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

        return z_mean, z_log_sigma_sq

    def _decoding_network(self, weights, biases):
        """Create decoding network"""
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        x_decode_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))

        return x_decode_mean

    def _create_loss_optimizer(self):
        reconstr_loss = \
            -tf.reduce_sum(self.vae_input*tf.log(1e-10 + self.x_decode_mean) +
                           (1-self.vae_input)*tf.log(1e-10 + 1-self.x_decode_mean), 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) -
                                           tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.vae_input: X})
        return cost

    def encode(self, X):
        # numpy_X = X.eval()
        return self.sess.run(self.z_mean, feed_dict={self.vae_input: X})

    def decode(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["code_size"])

        return self.sess.run(self.x_decode_mean, feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        return self.sess.run(self.x_decode_mean, feed_dict={self.vae_input: X})

    def save(self, filepath):
        np.save(file=filepath, arr=self.sess.run(self.network_weights))
        return 0.

    def load(self, filepath):
        weights = np.load(file=filepath).item()

    def load_weights(self, filepath):
        weights = np.load(file=filepath).item()
        # This made me as sad to write as it makes you to read
        assign = self.network_weights['encoder_weights']['h1'].assign(weights['encoder_weights']['h1'])
        self.sess.run(assign)
        assign = self.network_weights['encoder_weights']['h2'].assign(weights['encoder_weights']['h2'])
        self.sess.run(assign)
        assign = self.network_weights['encoder_weights']['out_mean'].assign(weights['encoder_weights']['out_mean'])
        self.sess.run(assign)
        assign = self.network_weights['encoder_weights']['out_log_sigma'].assign(weights['encoder_weights']['out_log_sigma'])
        self.sess.run(assign)

        assign = self.network_weights['encoder_biases']['b1'].assign(weights['encoder_biases']['b1'])
        self.sess.run(assign)
        assign = self.network_weights['encoder_biases']['b2'].assign(weights['encoder_biases']['b2'])
        self.sess.run(assign)
        assign = self.network_weights['encoder_biases']['out_mean'].assign(weights['encoder_biases']['out_mean'])
        self.sess.run(assign)
        assign = self.network_weights['encoder_biases']['out_log_sigma'].assign(
            weights['encoder_biases']['out_log_sigma'])
        self.sess.run(assign)

        assign = self.network_weights['decoder_weights']['h1'].assign(weights['decoder_weights']['h1'])
        self.sess.run(assign)
        assign = self.network_weights['decoder_weights']['h2'].assign(weights['decoder_weights']['h2'])
        self.sess.run(assign)
        assign = self.network_weights['decoder_weights']['out_mean'].assign(weights['decoder_weights']['out_mean'])
        self.sess.run(assign)
        assign = self.network_weights['decoder_weights']['out_log_sigma'].assign(weights['decoder_weights']['out_log_sigma'])
        self.sess.run(assign)

        assign = self.network_weights['decoder_biases']['b1'].assign(weights['decoder_biases']['b1'])
        self.sess.run(assign)
        assign = self.network_weights['decoder_biases']['b2'].assign(weights['decoder_biases']['b2'])
        self.sess.run(assign)
        assign = self.network_weights['decoder_biases']['out_mean'].assign(weights['decoder_biases']['out_mean'])
        self.sess.run(assign)
        assign = self.network_weights['decoder_biases']['out_log_sigma'].assign(
            weights['decoder_biases']['out_log_sigma'])
        self.sess.run(assign)
        # X = self.sess.run(self.network_weights)
        # print(X['encoder_biases']['out_mean'])

def train(network_architecture, learning_rate=0.0001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for ii in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size=batch_size,fake_data=False, shuffle=True)

            cost = vae.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae


def main():
    network_architecture = dict(n_hidden_recog_1=500, n_hidden_recog_2=500, n_hidden_gener_1=500, n_hidden_gener_2=500,
                                n_input=784, code_size=4)
    vae = train(network_architecture, training_epochs=100)
    # X = vae.sess.run(vae.network_weights)
    # print(X['encoder_biases']['out_mean'])
    # vae.load_weights('vae_weights.npy')
    # print(X)
    # print(X['weights_gener']['h1'])
    # saver = tf.train.Saver({'vae': vae.sess})
    # saver.save(vae.sess, 'vae_model')
    # saver.save(vae.sess, 'C:\\Users\\dgilton\\PycharmProjects\\WGANYO\\vae_model')
    vae.save(filepath='C:\\Users\\dgilton\\PycharmProjects\\WGANYO\\vae_weights_dim_4')
    # return

    x_sample = mnist.test.next_batch(100)[0]

    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    # canvas = np.empty((28 * ny, 28 * nx))
    # for i, yi in enumerate(x_values):
    #     for j, xi in enumerate(y_values):
    #         z_mu = np.array([[xi, yi]] * vae.batch_size)
    #         x_mean = vae.decode(z_mu)
    #         canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)
    #
    # plt.figure(figsize=(8, 10))
    # Xi, Yi = np.meshgrid(x_values, y_values)
    # plt.imshow(canvas, origin="upper", cmap="gray")
    # plt.tight_layout()
    # plt.show(block=True)

if __name__=="__main__":
    main()
