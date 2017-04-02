''' Feature selection using Chi-Square similarity
Uses naive_bayes.py
@author: Utkarsh Simha
'''
import nltk
import sklearn
import numpy as np
import cPickle as pickle

import naive_bayes as nb


def getVocab():
    return nb.getVocab()

def getReverseVocab( vocab ):
    rev_vocab = {}
    for idx, word in enumerate( vocab ):
        rev_vocab[ word ] = idx
    return rev_vocab


class FeatureSelection:
    def __init__( self, train, test, vocab, rev_vocab, bucket_data, M ):
        self.train = train
        self.test = test
        self.vocab = vocab
        self.rev_vocab = rev_vocab
        self.bucket_data = bucket_data
        self.M = M
        self.num_classes = num_classes = max( train[1] ) + 1
        self.total_docs = self.train[0].shape[0]
        self.chi_square = dict.fromkeys( vocab )

    def computeScore( self, word ):
        D = self.total_docs
        chi_square_per_class = []
        docs_of_class = []
        for c in range( self.num_classes ):
            class_docs = self.bucket_data[ c ]
            total_class_docs = len( class_docs )
            docs_with_t = map( lambda doc: word in doc, class_docs ).count( True )
            docs_without_t = total_class_docs - docs_with_t
            docs_of_class.append( ( docs_with_t, docs_without_t ) )

        for c in range( self.num_classes ):
            with_t = 0
            without_t = 1
            P = docs_of_class[ c ][ with_t ]
            Q = sum( zip( *docs_of_class )[with_t] ) - P
            M = docs_of_class[ c ][ without_t ]
            N = sum( zip( *docs_of_class )[without_t] ) - M
            try:
                score = ( D * ( P * N - M * Q )**2 ) / ( ( P + M ) * ( Q + N ) * ( P + Q ) * ( M + N ) )
            except:
                score = 0.0 
            chi_square_per_class.append( score )

        self.chi_square[ word ] = chi_square_per_class

    def getIndex( self, word ):
        return self.rev_vocab[ word ]

    def getWord( self, index ):
        return self.vocab[ word ]

    def calcChiSquare( self ):
        for word in self.vocab:
            self.computeScore( word )
        print "Finished computing Chi Square"

def getData():
    train,test = nb.getData()
    return (train,test) 

def getBucketData( train ):
    num_classes = max( train[1] ) + 1
    return nb.divideIntoClasses( train, num_classes )

if __name__ == '__main__':
    M = 5000
    train, test = getData()
    print "Got data"
    vocab = getVocab()
    print "Got vocab with size: {}".format( len( vocab ) )
    rev_vocab = getReverseVocab( vocab )
    bucket_data = getBucketData( train )
    print "Got Bucket data" 
    feat_select = FeatureSelection( train, test, vocab, rev_vocab, bucket_data, M )
    "Starting Chi Square computation.."
    feat_select.calcChiSquare()

    "Pickling.."
    pickle.dump( feat_select.chi_square, open( "chi_square.p", "wb" ) )
