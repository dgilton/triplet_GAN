import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

np.random.seed(0)
tf.set_random_seed(0)

mnist = read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

def xavier_init(fan_in, fan_out, constant=1):
    """Initialize network weights with Xavier Initialization"""
    low = -constant*np.sqrt(6/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# -0.7361542, -1.64742219
# I assume we're doing MNIST data here
transfer_fct=tf.nn.softplus
learning_rate = 0.0001
batch_size = 100

input_size            = 28*28
encoder_hlayer_1_size = 500
encoder_hlayer_2_size = 500
code_length           = 5
decoder_hlayer_1_size = 500
decoder_hlayer_2_size = 500


vae_input = tf.placeholder(tf.float32, [None, input_size])


# Drawing lines in a graph
enc_hlayer_1_weights = tf.Variable(xavier_init(input_size, encoder_hlayer_1_size))
enc_hlayer_1_bias = tf.Variable(tf.zeros([encoder_hlayer_1_size]), dtype=tf.float32)

enc_hlayer_2_weights = tf.Variable(xavier_init(encoder_hlayer_1_size, encoder_hlayer_2_size))
enc_hlayer_2_bias = tf.Variable(tf.zeros([encoder_hlayer_2_size]), dtype=tf.float32)

code_mean_weights = tf.Variable(xavier_init(encoder_hlayer_2_size, code_length))
code_logsigma_weights = tf.Variable(xavier_init(encoder_hlayer_2_size, code_length))

code_mean_bias = tf.Variable(tf.zeros([code_length]), dtype=tf.float32)
code_logsigma_bias = tf.Variable(tf.zeros([code_length]), dtype=tf.float32)

dec_hlayer_1_weights = tf.Variable(xavier_init(code_length, decoder_hlayer_1_size))
dec_hlayer_1_bias = tf.Variable(tf.zeros([decoder_hlayer_1_size]), dtype=tf.float32)

dec_hlayer_2_weights = tf.Variable(xavier_init(decoder_hlayer_1_size, decoder_hlayer_2_size))
dec_hlayer_2_bias = tf.Variable(tf.zeros([decoder_hlayer_2_size]), dtype=tf.float32)

output_mean_weights = tf.Variable(xavier_init(decoder_hlayer_2_size, input_size))
output_logsigma_weights = tf.Variable(xavier_init(decoder_hlayer_2_size, input_size))

output_mean_bias = tf.Variable(tf.zeros([input_size]), dtype=tf.float32)
output_logsigma_bias = tf.Variable(tf.zeros([input_size]), dtype=tf.float32)


def encoding_net(input_data):
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(input_data, enc_hlayer_1_weights), enc_hlayer_1_bias))
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, enc_hlayer_2_weights), enc_hlayer_2_bias))

    z_mean = tf.add(tf.matmul(layer_2, code_mean_weights), code_mean_bias)
    z_logsigma_sq = tf.add(tf.matmul(layer_2, code_logsigma_weights), code_logsigma_bias)
    return z_mean, z_logsigma_sq

def decoding_net(code):
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(code, dec_hlayer_1_weights), dec_hlayer_1_bias))
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, dec_hlayer_2_weights), dec_hlayer_2_bias))

    decoded_sig = tf.add(tf.matmul(layer_2, output_mean_weights), output_mean_bias)
    decoded_decision = tf.nn.sigmoid(decoded_sig)
    return decoded_decision


gaussian_mean, gaussian_logsigma_sq = encoding_net(vae_input)

epsilon = tf.random_normal((batch_size, code_length), 0, 1, dtype=tf.float32)
z = tf.add(gaussian_mean, tf.multiply(tf.sqrt(tf.exp(gaussian_logsigma_sq)), epsilon))

decoded_signal = decoding_net(z)


reconstr_loss = -tf.reduce_sum(vae_input*tf.log(1e-10 + decoded_signal) +
                               (1-vae_input)*tf.log(1e-10 + 1 - decoded_signal))
latent_loss = -0.5 * tf.reduce_sum(1 + gaussian_logsigma_sq - tf.square(gaussian_mean) -
                                   tf.exp(gaussian_logsigma_sq), 1)
loss = tf.reduce_mean(reconstr_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



def main():

    training_epochs = 10
    display_steps = 2

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for ii in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size=batch_size,fake_data=False, shuffle=True)

            _, cost, lossy = sess.run(
                [optimizer, loss, latent_loss],
                feed_dict={vae_input: batch_xs}
                )
            avg_cost += cost / n_samples * batch_size

        if epoch % display_steps == 0:
            print(np.mean(lossy))
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))






if __name__ == "__main__":
    main()
