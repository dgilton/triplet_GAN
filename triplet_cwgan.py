import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import VAE

# We defining global constants, boys.
sqrt3 = np.sqrt(3)

mb_size = 32
top_dim = 784
second_dim = int(round(top_dim))         # Arbitrary but it worked
third_dim = int(round(second_dim/3))     # Also arbitrary, also worked
noise_dim = 50                           # ^^^^^^^^^^^^^^^^^^^^^^^^^^^
h_dim = 100                              # ^^^^^^^^^^^^^^^^^^^^^^^^^^^
lam = 10
discriminator_training_rounds = 5
lr = 1e-4

code_dim = 4

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
label_dim = 2*code_dim


def plot(samples):
    fig = plt.figure(figsize=(3, 3))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.05, hspace=0.05)

    for ii, sample in enumerate(samples):
        ax = plt.subplot(gs[ii])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def STE_loss(xi, xj, xl, alpha):
    xi_xj_distance = tf.reduce_sum(tf.square(tf.subtract(xi, xj)), axis=1)
    xi_xl_distance = tf.reduce_sum(tf.square(tf.subtract(xi, xl)), axis=1)
    pijl_exponent = -(alpha+1.0)/2.0
    numerator = (1.0 + xi_xj_distance/alpha)**pijl_exponent
    denominator = (1 + xi_xj_distance/alpha)**pijl_exponent + (1+xi_xl_distance/alpha)**pijl_exponent
    return numerator/denominator


X = tf.placeholder(tf.float32, shape=[None, top_dim])
y = tf.placeholder(tf.float32, shape=[None, label_dim])
X_j = tf.placeholder(tf.float32, shape=[None, code_dim])
X_k = tf.placeholder(tf.float32, shape=[None, code_dim])
X_i = tf.placeholder(tf.float32, shape=[None, code_dim])

D_W1 = tf.Variable(xavier_init([top_dim + label_dim, second_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[second_dim]))

D_W2 = tf.Variable(xavier_init([second_dim, third_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[third_dim]))

D_W3 = tf.Variable(xavier_init([third_dim, h_dim]))
D_b3 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W4 = tf.Variable(xavier_init([h_dim, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]


noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])

G_W1 = tf.Variable(xavier_init([noise_dim + label_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, third_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[third_dim]))

G_W3 = tf.Variable(xavier_init([third_dim, second_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[second_dim]))

G_W4 = tf.Variable(xavier_init([second_dim, top_dim]))
G_b4 = tf.Variable(tf.zeros(shape=[top_dim]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator_net(noise_input, y):
    input = tf.concat(axis=1, values=[noise_input,y])
    G_h1 = tf.nn.relu(tf.matmul(input, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)

    G_log_prob = tf.matmul(G_h3, G_W4) + G_b4
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator_net(X, y):
    input = tf.concat(axis=1, values=[X,y])
    # input = X
    D_h1 = tf.nn.relu(tf.matmul(input, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_logit = tf.matmul(D_h3, D_W4) + D_b4
    return D_logit


G_sample = generator_net(noise_input, y)
D_real = discriminator_net(X, y)
D_fake = discriminator_net(G_sample, y)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
# y_inter = np.zeros(shape=[mb_size,label_dim])
# y_inter[:,7] = 1
# y_inter = y
grad = tf.gradients(discriminator_net(X_inter, y), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum(grad**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)


STE = STE_loss(X_i, X_j, X_k, code_dim-1.0)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake) - tf.reduce_mean(STE)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G))

vae_architecture = dict(n_hidden_recog_1=500, n_hidden_recog_2=500, n_hidden_gener_1=500, n_hidden_gener_2=500,
                        n_input=784, code_size=code_dim)

vae = VAE.VariationalAutoencoder(network_architecture=vae_architecture)
vae.load_weights(filepath='vae_weights_dim_4.npy')

sess = tf.Session()
sess.run(tf.global_variables_initializer())


if not os.path.exists('outTrip/'):
    os.makedirs('outTrip/')

# Keep track of how many thousands of iterations we've done
i = 0

for iterator in range(1000000):
    y_mb = np.zeros((mb_size, 2*code_dim))
    for _ in range(discriminator_training_rounds):
        X_mb, _ = mnist.train.next_batch(mb_size)

        embedding_coords = vae.encode(X_mb)

        permutation_0 = np.arange(mb_size)
        np.random.shuffle(permutation_0)

        permutation_1 = np.arange(mb_size)
        np.random.shuffle(permutation_1)

        sample_xj_storage = np.ndarray(shape=np.shape(X_mb), dtype=float)
        sample_xk_storage = np.ndarray(shape=np.shape(X_mb), dtype=float)
        Xj = np.ndarray(shape=np.shape(embedding_coords), dtype=float)
        Xk = np.ndarray(shape=np.shape(embedding_coords), dtype=float)
        for ii in range(mb_size):
            xi_coord = embedding_coords[ii, :]
            xj_coord = embedding_coords[permutation_0[ii], :]
            xk_coord = embedding_coords[permutation_1[ii], :]
            if np.linalg.norm(xi_coord - xj_coord) > np.linalg.norm(xi_coord - xk_coord):
                Xj[ii, :] = xk_coord
                Xk[ii, :] = xj_coord

                sample_xj_storage[ii, :] = X_mb[permutation_1[ii], :]
                sample_xk_storage[ii, :] = X_mb[permutation_0[ii], :]
            else:
                Xj[ii, :] = xj_coord
                Xk[ii, :] = xk_coord

                sample_xj_storage[ii, :] = X_mb[permutation_0[ii], :]
                sample_xk_storage[ii, :] = X_mb[permutation_1[ii], :]

        y_mb = np.concatenate((Xj, Xk), axis=1)

        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_mb, noise_input: sample_z(mb_size, noise_dim), y: y_mb, X_j: Xj, X_k: Xk}
        )

    # If you can figure out a way to feed the VAE the G_sample tensor, please change this. As is, the triplet loss is
    # formed from different samples than the standard GAN loss, which feels bad. Should be the same network, but still.
    generator_sample = sess.run(G_sample, feed_dict={noise_input: sample_z(mb_size, noise_dim), y: y_mb})
    sampled_coordinates = vae.encode(generator_sample)

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={noise_input: sample_z(mb_size, noise_dim), y: y_mb, X_j: Xj, X_k: Xk, X_i: sampled_coordinates}
    )

    if iterator % 1000 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(iterator, D_loss_curr, G_loss_curr))

        if iterator % 1000 == 0:
            n_sample = 32
            noise_sample = sample_z(n_sample, noise_dim)

            samples = sess.run(G_sample, feed_dict={noise_input: noise_sample, y: y_mb[range(n_sample), :]})

            samples[3, :] = samples[1, :]
            samples[6, :] = samples[2, :]
            samples[1, :] = sample_xj_storage[0, :]
            samples[2, :] = sample_xk_storage[0, :]
            samples[4, :] = sample_xj_storage[1, :]
            samples[5, :] = sample_xk_storage[1, :]
            samples[7, :] = sample_xj_storage[2, :]
            samples[8, :] = sample_xk_storage[2, :]

            i_coord = vae.encode(samples)
            # ij_dist = np.linalg.norm(i_coord-Xj)
            # ik_dist = np.linalg.norm(i_coord-Xk)

            for it_thu_samples in range(3):
                ij_dist = np.linalg.norm(i_coord[it_thu_samples, :] - Xj[it_thu_samples, :])
                ik_dist = np.linalg.norm(i_coord[it_thu_samples, :] - Xk[it_thu_samples, :])
                print(ij_dist)
                print(ik_dist)

            fig = plot(samples[0:9])
            plt.savefig('outTrip/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)