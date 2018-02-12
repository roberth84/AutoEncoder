import os, shutil
import functools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import time

PWD = os.getcwd()
LOGDIR = PWD + '/log/'


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:

    def __init__(self, x, encoding_dim = 32):

        self.x = x
        self.encoding_dim = encoding_dim
        self.embedding_size = encoding_dim
        
        self.encoder
        self.decoder
        self.decoder_sigmoid
        self.loss
        self.train
        self.summ
        self.embedding_input

    @define_scope
    def encoder(self):

        w = tf.get_variable("W", [784, self.encoding_dim],
                            initializer = tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable("B", [self.encoding_dim],
                            initializer = tf.zeros_initializer())

        act = tf.nn.relu(tf.add(tf.matmul(self.x, w), b))

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        return act
        

    @define_scope
    def decoder(self):

        initializer = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable("W", shape = [self.encoding_dim, 784],
                            initializer = tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable("B", [784], initializer = tf.zeros_initializer())
        
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)

        val = tf.add(tf.matmul(self.encoder, w), b)

        return val
        

    @define_scope
    def decoder_sigmoid(self):
        
        return tf.nn.sigmoid(self.decoder)

    
    @define_scope
    def loss(self):

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.decoder, labels=self.x),
            name='loss')

        tf.summary.scalar("loss", loss)

        return loss
    
    @define_scope
    def train(self):

        return tf.train.AdadeltaOptimizer(
            learning_rate=1.0, rho=0.95, epsilon=1e-07).minimize(self.loss)

    @property
    def embedding_input(self):

        # embeddings for TensorBoard projector
        self.embedding_size = self.encoding_dim
        return self.encoder
    
    @property
    def summ(self):
        return tf.summary.merge_all()
    
def main():

    tf.reset_default_graph() # must be called outside of tf.Session
    sess = tf.Session()

    # get MNIST images
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir = PWD + '/data',
                                                           one_hot = True)

    # placeholder for image
    x = tf.placeholder(tf.float32, shape = [None, 784], name = 'x')

    # build model
    model = Model(x, encoding_dim = 32)

    init = tf.global_variables_initializer()
    
    sess.run(init)

    embedding_var = tf.Variable(tf.zeros([1024,model.embedding_size]), name="test_embedding")
    assignment = embedding_var.assign(model.embedding_input)
    saver = tf.train.Saver()
    
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.sprite.image_path = PWD + '/sprite_1024.png'
    embedding.metadata_path = PWD + '/labels_1024.tsv'

    # Specify the width and height of a single thumbnail.
    embedding.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
                                                               
    nepochs = 5
    if False:
        niter = 10
        batchsize = 10
        log_niter = 1
    else:
        niter = 215
        batchsize = 256
        log_niter = 22

    for epoch in range(nepochs):

        for i in range(niter):

            batch = mnist.train.next_batch(batchsize)

            # logging for TensorBoard
            if i % log_niter == 0:
                s = sess.run(model.summ, feed_dict={x: batch[0]})
                writer.add_summary(s, i)

            # train model, print loss at end of each epoch
            _, l = sess.run([model.train, model.loss], feed_dict={x: batch[0]})
            if i == niter - 1:
                print('Epoch %i: Minibatch Loss: %f' % (epoch, l))
                
                sess.run(assignment, feed_dict={x: mnist.test.images[:1024]})
                saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)

    print("Done")

    # run over test images and plot input/output        
    x_test = mnist.test.images

    decoded_output = sess.run(model.decoder_sigmoid, feed_dict={x: x_test})

    n = 10 # number of images 
    plt.figure(figsize=(20,4))
    for i in range(n):
    
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_output[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()

if __name__ == '__main__':
    main()
