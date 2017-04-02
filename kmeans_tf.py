''' KMeans clustering on MNIST using TensorFlow
@author: Utkarsh Simha 
'''
import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import cPickle
import gzip
import matplotlib.pyplot as plt
import time

def loadData( mnist_fpath ):
    with gzip.open(mnist_fpath, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    train_set = train_set[0], train_set[1].reshape( train_set[1].shape[0], 1 )
    test_set = test_set[0], test_set[1].reshape( test_set[1].shape[0], 1 )
    return train_set, test_set

class Clustering:
    def __init__( self, num_samples, num_feats, num_clusters ):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.num_feats = num_feats
            self.num_clusters = num_clusters
            self.num_samples = num_samples
            self.train_x = tf.placeholder( tf.float32, shape=( None, self.num_feats ) )

            self.cluster_assign = tf.Variable( tf.zeros( [ self.num_samples ], dtype=tf.int32 ) )
            self.centroids = tf.Variable(tf.slice(tf.random_shuffle(self.train_x), [0, 0], [self.num_clusters, -1]))

            expanded_centroids = tf.expand_dims( self.centroids, 1 )
            expanded_data = tf.expand_dims( self.train_x, 0 )

            local_dist = tf.reduce_sum( tf.square( expanded_centroids - expanded_data ), 2 )
            self.cent_assignments = tf.argmin( local_dist, 1 )

            self.means = []
            for c in range( self.num_clusters ):
                mean = tf.reduce_mean(
                        tf.gather( self.train_x, tf.reshape( tf.where(
                            tf.equal( self.cent_assignments, c )
                            ), [1,-1] ) ), 1
                        )
                self.means.append( mean )

            recalc_centroids = tf.concat( 0, self.means )
            self.update_centroids = tf.assign( self.centroids, recalc_centroids )

if __name__ == '__main__':
    mnist_fpath = "./mnist.pkl.gz"
    ( train_x, train_y ), ( test_x, test_y ) = loadData( mnist_fpath )
    train_x = train_x[:10000]
    n_clusters = 100
    clust = Clustering( train_x.shape[0], train_x.shape[1], n_clusters )
    feed_dict = { clust.train_x: train_x }
    num_iters = 10
    with tf.Session( graph = clust.graph ) as sess:
        tf.initialize_all_variables().run( feed_dict=feed_dict )
        for itr in range( num_iters ):
            res = sess.run( [ clust.centroids, clust.means, clust.cent_assignments ], feed_dict=feed_dict )
