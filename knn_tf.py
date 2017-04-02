''' K-Nearest Neigbhor algorithm on MNIST using TensorFlow 
@author: Utkarsh Simha
'''
import tensorflow as tf
import numpy as np
import cPickle
import gzip

def loadData( mnist_fpath ):
    with gzip.open(mnist_fpath, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    train_set = train_set[0], train_set[1].reshape( train_set[1].shape[0], 1 )
    test_set = test_set[0], test_set[1].reshape( test_set[1].shape[0], 1 )
    return train_set, test_set

class KNearestNeighbours:

    def __init__( self, k ):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.k = k
            self.train_inp = tf.placeholder( tf.float32, shape=( None, 784 ) )
            self.train_out = tf.placeholder( tf.int32, shape=( None, 1 ) )
            self.test_inp = tf.placeholder( tf.float32, shape=( None, 784 ) )
            self.test_out = tf.placeholder( tf.int32, shape=( None,1 ) )
            
            self.dist = tf.sqrt( tf.reduce_sum( tf.square( tf.sub( tf.expand_dims( self.train_inp, 1 ), self.test_inp ) ), 1 ) ) 


def accuracy( pred, labels ):
    correct = 0.0
    for i,j in zip( pred, labels ):
        if( i == j ):
            correct += 1
    return ( correct/len(pred) ) * 100

if __name__ == '__main__':
    mnist_fpath = "./mnist.pkl.gz"
    ( train_x, train_y ), ( test_x, test_y ) = loadData( mnist_fpath )
    train_x = train_x[:5000]
    train_y = train_y[:5000]
    test_x = test_x[:1000]
    test_y = test_y[:1000]
    print "Shapes of input :"
    print "Training:   ",train_x.shape, train_y.shape
    print "Test:       ",test_x.shape, test_y.shape
    knn = KNearestNeighbours( 1 )
    with tf.Session( graph = knn.graph, config = config ) as session:
        tf.initialize_all_variables().run()
        correct_classified = 0.0
        feed_dict = { knn.train_inp : train_x, knn.train_out : train_y,
                        knn.test_inp : test_x, knn.test_out : test_y } 
        dist = session.run( knn.dist, feed_dict=feed_dict )
        print accuracy( pred[0], test_y )
