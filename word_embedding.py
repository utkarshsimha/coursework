''' Word embeddings using Point-wise mutual information 
and Spectral Clustering
@author: Utkarsh Simha
'''
import nltk
from nltk.corpus import brown 
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import itertools
import numpy as np
import cPickle as pickle
import sklearn.cluster
import sklearn.manifold
from sklearn.neighbors import NearestNeighbors

def preprocess( words ):
    words = [ i.lower() for i in words if i.lower() not in stop ]
    tokenizer = RegexpTokenizer( r'\w\w+' )
    words = tokenizer.tokenize( ' '.join( words ) )
    return words

class WordEmbedding:
    def __init__( self, words, win_size=2 ):
        self.words = words
        self.win_size = win_size
        self.fdist = nltk.FreqDist( words )
        self.V, self.V_freq = zip( *self.fdist.most_common(5000) )
        pickle.dump( self.V, open( "vocab.p", "wb" ) )
        self.C, self.C_freq = zip( *self.fdist.most_common(1000) )
        self.get_context()
        self.calc_PMI()
        pickle.dump( self.phi, open( "phi.p", "wb" ) )
        self.embed_word_vecs( self.phi )
        pickle.dump( self.embeddings, open( "embeddings.p", "wb" ) )
        self.cluster_embeddings( self.embeddings, 100 )
        pickle.dump( self.clustered_words, open( "clustered_words.p", "wb" ) )
        
    def get_context( self ):
        self.context = {}
        concordance = nltk.ConcordanceIndex( self.words )
        for w in self.V:
            offsets = concordance.offsets( w )
            self.context[ w ] = nltk.FreqDist( itertools.chain.from_iterable( [ self.words[offset-self.win_size:offset] \
                    + self.words[ offset+1:offset+1+self.win_size ] for offset in offsets ] ) )

    def calc_PMI( self ):
        self.phi = np.zeros( ( len(self.V), len(self.C) ) )
        n_c_w = np.zeros( ( len(self.V), len(self.C) ), dtype=np.float32 )
        p_c = np.asarray( self.C_freq )/ float( np.sum(self.C_freq) )
        for v, v_freq in zip( enumerate( self.V ), self.V_freq ):
            v_idx, v = v
            for c, c_freq in zip( enumerate( self.C ), self.C_freq ):
                c_idx, c = c
                n_c_w[ v_idx, c_idx ] = self.context[ v ][ c ]
            self.phi[ v_idx ] = np.log( ( n_c_w[ v_idx ]/float(v_freq) ) / p_c  )
            self.phi[ v_idx ][ self.phi[ v_idx ] < 0 ] = 0.0

    def embed_word_vecs( self, word_vecs ):
        se = sklearn.manifold.SpectralEmbedding( n_components=100 )
        self.embeddings = se.fit_transform( word_vecs )

    def cluster_embeddings( self, embeddings, n_clusters ):
        kmeans = sklearn.cluster.KMeans( n_clusters=n_clusters, random_state=0 )
        kmeans.fit( embeddings )
        self.clustered_words = [ np.asarray( self.V )[ np.where( kmeans.labels_ == i ) ] for i in range( n_clusters ) ]

    def get_nearest_neighbor( self, test_words ):
        test_word_indices = np.random.randint( 0, 5000, ( 25, ) )
        test_word_indices = [ list(self.V).index( w ) for w in test_words ]
        nn_indices = []
        for w in test_word_indices:
             nn_indices.append( np.argmin( [ 1 - ( np.dot( self.embeddings[ w ], self.embeddings[ w_ ] ) \
                    / ( np.linalg.norm( self.embeddings[ w ] ) * np.linalg.norm( self.embeddings[ w_ ] ) ) ) for w_ in range( 5000 ) if w!= w_ ] ) )
        return zip( np.asarray( self.V )[ test_word_indices ], np.asarray( self.V )[ nn_indices ] )

if __name__ == '__main__':
    words = brown.words()
    words = preprocess( words )
    word_embedding = WordEmbedding( words )
    test_words = "communism,state,specialists,arrested,respective,companion,businesses,racial".strip().split(",")
    print word_embedding.get_nearest_neighbor( test_words )


