
import numpy as np
import tensorflow as tf

class EmbedLoader(object):

    def __init__(self):
        
        """ Initializing word_embedding with W2V """

        self.pre_emb = dict()
        self.tokenToIndexMap = {}
    
    def loadW2V(self, emb_path, type = "bin"):
        num_keys = 0
        if type == "text":
            for line in open(emb_path):
                l = line.strip().split()
                st = l[0].lower()
                self.pre_emb[st] = np.asarray([l[1:]])

    def getLocalIndexForVocab(self, word):
        return self.tokenToIndexMap.get(word, "NOT FOUND")

    def loadVocabWithIndex(self, processed_dir):
        wordToIndexFile = processed_dir + '/wordToIndex.csv'
        print(wordToIndexFile)
        with tf.gfile.Open(wordToIndexFile, 'r') as f:
            for line in f.readlines():
                values = line.strip().split(",")
                self.tokenToIndexMap[values[0]] = values[1]
        print("tokenToIndexMap ", self.tokenToIndexMap)

    def getVocab(self, word):
        return (self.pre_emb[word])
        
    def createTemporaryEmbedding(self, vocab_size, word_embed_size):
        
        #initW = tf.get_variable('word_embedding', [vocab_size, word_embed_size],initializer=tf.random_uniform_initializer(-0.25,0.25))

        print("vocab_size ", vocab_size, " word_embed_size ", word_embed_size)
        
        initW = np.random.uniform(-0.25,0.25,(vocab_size, word_embed_size))
        
        for key_, index in self.tokenToIndexMap.items():
            try:
                key = (key_.replace("_", ""))
                val = self.getVocab(key).ravel()
                #print("val shape ", val.shape)
                initW[int(index)] = val
            except KeyError:
                pass
                #print("key not found ", key)

        #print("initW ", initW[1813])
        return tf.convert_to_tensor(initW, dtype=tf.float32)
        #return initW