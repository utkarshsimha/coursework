''' Co-ordinate descent for Logistic regression on the Wine Dataset
using TensorFlow
@author: Utkarsh Simha
'''
import numpy as np
import tensorflow as tf
import cPickle as pickle
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import sys
from sklearn_log_reg import sklearnLogReg
import matplotlib.pyplot as plt

def printer( string ):
    sys.stdout.write("\r\x1b[K"+string. __str__())
    sys.stdout.flush()

def loadDataset( split=128 ):
    inp, targ = pickle.load( open( "wine_data.p", "rb" ) )
    inp, targ = shuffle( inp, targ )
    train = ( inp[:split], targ[:split] )
    test = ( inp[split:], targ[split:] )
    return (train, test)

class LogisticRegression:
    def __init__( self, train, test,  batch_size=128, random=False, optimizer="adagrad" ):
        self.graph = tf.Graph()
        with self.graph.as_default():
            n_in = train[0].shape[1]
            n_out = train[1].shape[1]
            self.train_X = tf.placeholder( tf.float32, shape=[ batch_size, n_in ], name="train_X" )
            self.train_Y = tf.placeholder( tf.float32, shape=[ batch_size, n_out ], name="train_Y" )
            self.learning_rate = tf.placeholder( tf.float32, shape=(), name="learning_rate" )
            test_X = tf.constant( test[0], dtype=tf.float32, name="test_X" )
            self.W = tf.Variable( tf.truncated_normal( [ n_in, n_out ] ), name="W" )

            with tf.name_scope("train_out"):
                train_out = tf.nn.softmax( tf.matmul( self.train_X, self.W ) )
            with tf.name_scope("test_out"):
                test_out = tf.nn.softmax( tf.matmul( test_X, self.W ) )
            with tf.name_scope("loss"):
                self.loss = tf.contrib.losses.log_loss( train_out, self.train_Y )
            with tf.name_scope("error"):
                self.error = tf.reduce_mean( tf.expand_dims( tf.cast( self.train_Y - train_out, tf.float32 ), 1 )\
                        * tf.expand_dims( self.train_X, 2 ), 0 )
            if random is False:
                indices = tf.where( tf.equal( self.error, tf.reduce_max( self.error, [0,1] ) ) )[0]
                x_cord = tf.cast( indices[0], tf.int32 )
                y_cord = tf.cast( indices[1], tf.int32 )
            else:
                x_cord = tf.to_int32( tf.abs( tf.random_normal( () ) * 1000 % n_in ) ) 
                y_cord = tf.to_int32( tf.abs( tf.random_normal( () ) * 1000 % n_out ) ) 
            self.cord = ( x_cord, y_cord )

            if( optimizer == 'co-ord' ):
                with tf.variable_scope("cord_descent"):
                    self.optimizer = None
                    self.grad = self.error[ x_cord ][ y_cord ]
                    delta = tf.SparseTensor( indices=[ [ tf.cast( x_cord, tf.int64 ), tf.cast( y_cord, tf.int64 ) ]],\
                            values=self.grad, dense_shape=[ n_in, n_out ] )
                    self.W = tf.assign_add( self.W, self.learning_rate * tf.sparse_tensor_to_dense( delta ) )
            elif( optimizer == 'adagrad' ):
                self.optimizer = tf.train.AdagradOptimizer( self.learning_rate ).minimize( self.loss )

            elif( optimizer == 'sgd' ):
                self.optimizer = tf.train.GradientDescentOptimizer( self.learning_rate ).minimize( self.loss )

            with tf.name_scope("train_acc"):
                self.train_acc = tf.equal(tf.argmax(train_out, 1), tf.argmax(train_Y, 1))
                self.train_acc = tf.reduce_mean(tf.cast(self.train_acc, tf.float32)) * 100
            with tf.name_scope("test_acc"):
                self.test_acc = tf.equal(tf.argmax(test_out, 1), tf.argmax(test_Y, 1))
                self.test_acc = tf.reduce_mean(tf.cast(self.test_acc, tf.float32)) * 100

            tf.scalar_summary( "loss", self.loss )
            tf.scalar_summary( "train_acc", self.loss )
            tf.scalar_summary( "test_acc", self.loss )
            tf.histogram_summary( "weights", self.W )
            tf.histogram_summary( "error", self.error )
            self.merged_summaries = tf.merge_all_summaries()

if __name__ == '__main__':
    batch_size = 128
    n_epochs = 150000
    split = 128
    train, test = loadDataset( split )
    print 'Training set', train[0].shape, train[1].shape
    print 'Test set', test[0].shape, test[1].shape
    train_X, train_Y = train
    test_X, test_Y = test
    ones = np.ones( ( train_X.shape[0], 1 ) )
    train_X = sklearn.preprocessing.normalize( np.hstack( ( train_X, ones ) ) )
    ones = np.ones( ( test_X.shape[0], 1 ) )
    test_X = sklearn.preprocessing.normalize( np.hstack( ( test_X, ones ) ) )
    train = ( train_X, train_Y )
    test = ( test_X, test_Y )

    lr = LogisticRegression( train, test, batch_size=batch_size, random=True, optimizer='co-ord' )
    costs = []
    train_accuracies = []
    test_accuracies = []
    with tf.Session( graph = lr.graph ) as session:
        summary_writer = tf.train.SummaryWriter("/tmp/tf_logs/log_reg_rand", graph=lr.graph)
        feed_dict = { lr.train_X:train_X, lr.train_Y:train_Y }
        session.run( tf.initialize_all_variables(), feed_dict=feed_dict )
        eta_0 = 0.1
        for ep in range( 1, n_epochs ):
            eta = eta_0 / ( 1 + 0.00001 * (ep+1) )
            feed_dict = { lr.train_X:train_X, lr.train_Y:train_Y, lr.learning_rate: eta }
            summaries, cost = session.run( [ lr.merged_summaries, lr.loss ], feed_dict=feed_dict )
            if( lr.optimizer is not None ):
                _ = session.run( lr.optimizer, feed_dict=feed_dict )
            cost = session.run( lr.loss, feed_dict=feed_dict )
            cord = session.run( lr.cord, feed_dict=feed_dict )
            grad = session.run( lr.grad, feed_dict=feed_dict )
            W = session.run( lr.W, feed_dict=feed_dict )
            lrate = session.run( lr.learning_rate, feed_dict=feed_dict )
            err = session.run( lr.error, feed_dict=feed_dict )
            train_acc = session.run( lr.train_acc, feed_dict=feed_dict )
            test_acc = session.run( lr.test_acc, feed_dict=feed_dict )
            costs.append(cost)
            train_accuracies.append( train_acc )
            test_accuracies.append( test_acc )
            summary_writer.add_summary( summaries, ep )
            if( ep % 100 == 0 ):
                printer("Cost at {} - {}".format( ep, cost ) +\
                        " | Training accuracy : {} | cord : {} | grad : {} eta : {}".format( train_acc, cord, grad, lrate ) )

        print "\nTest accuracy : {}".format( test_acc )
