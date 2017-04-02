''' Implementation of Multinomial Naive Bayes Classifier
on the 20 Newsgroup dataset
@author: Utkarsh Simha
'''

import nltk
import sklearn
import numpy as np
import unicodedata
import itertools
import time
import cPickle as pickle
import sys

from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer


def preProcessText( data ):
    tokenizer = RegexpTokenizer(r'\w+')
    clean_doc = lambda doc: ' '.join( tokenizer.tokenize( doc ) ).encode( 'ascii', 'ignore' ).lower().split()
    processed_data = [ clean_doc( doc ) for doc in data ]
    return processed_data

def getData():
    train = newsgroups_train = fetch_20newsgroups(subset='train')
    test = newsgroups_train = fetch_20newsgroups(subset='test')
    return ( np.array( preProcessText( train.data ) ), train.target ), ( np.array( preProcessText( test.data ) ), test.target )

def getVocab():
    with open( "vocabulary.txt", "r" ) as f:
        vocab = f.read().split('\n')
        vocab.pop()
        return vocab

def divideIntoClasses( train, num_classes ):
    bucket_data = [ None ] * num_classes
    data, target = train
    for c in range( num_classes ):
        bucket_data[ c ] = data[ np.where( target == c )[0] ]
    return np.array( bucket_data )

def computePrior( bucket_data, target, num_classes ):
    prior = [ 0.0 ] * num_classes 
    total_docs = float( target.shape[0] )
    for c in range( num_classes ):
        prior[ c ] = bucket_data[ c ].shape[0] / total_docs 
    return prior

def computeConditional( bucket_data, vocab, num_classes ):
    cond_probs = {}
    concat_data = [ None ] * num_classes
    for c, class_data in enumerate( bucket_data ):
        concat_data[ c ] = list( itertools.chain.from_iterable( class_data ) )

    for word in vocab:
        cond_probs[ word ] = []
        for c in range( num_classes ):
            cond_probs[ word ].append( ( concat_data[ c ].count( word ) + 1 ) / ( float( len( concat_data[ c ] ) ) + len( vocab ) ) )
    return cond_probs

def classifyDoc( doc, prior, cond_probs, vocab, num_classes ):
    post_probs = []
    for c in range( num_classes ):
        prob_prod = prior[ c ]
        for token in doc:
            try:
                p_t_given_c = cond_probs[ token ][ c ]
            except KeyError:
                continue
            prob_prod *= p_t_given_c
        post_probs.append( prob_prod )
    target_class = np.argmax( post_probs )
    return target_class



if __name__ == '__main__':
    f_name = ""
    if( len( sys.argv ) == 2  ):
        f_name = sys.argv[1] 
    else:
        print "Provide vocab file name"
        exit()
    train, test = getData()
    vocab = getVocab()
    '''np.random.shuffle( vocab )
    np.random.shuffle( vocab )
    np.random.shuffle( vocab )
    vocab = vocab[:20000] '''
    #vocab = pickle.load( open( f_name, "rb" ) )
    print "Vocab size: {}".format( len( vocab ) )
    num_classes = max( train[1] ) + 1
    bucket_data = divideIntoClasses( train, num_classes )
    prior = computePrior( bucket_data, train[1], num_classes )
    cond_probs = computeConditional( bucket_data, vocab, num_classes )
    correct = 0.0
    count = 0.0
    for data, target in zip( test[0], test[1] ):
        count += 1
        pred = classifyDoc( data, prior, cond_probs, vocab, num_classes )
        actual = target
        if( pred == target ):
            correct += 1
    print 100 * ( correct / count )
