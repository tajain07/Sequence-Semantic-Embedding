
import numpy as np

class EmbedLoader(object):
    pre_emb = dict()
    vocab_size = 0
    word_embed_size = 300
    initW = None
    tokenToIndexMap = {}
    
    def loadW2V(self, emb_path, type = "bin"):
        num_keys = 0
        if type == "text":
            for line in open(emb_path):
                l = line.strip().split()
                st = l[0].lower()
                self.pre_emb[st] = np.asarray([l[1:]])

    
    def getVocab(self, word):
        return self.pre_emb[word]

    def createTemporaryEmbedding(self, vocab_size, word_embed_size):
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.initW = np.random.uniform(-0.25,0.25,(self.vocab_size, self.word_embed_size))
        print("self.initW ", self.initW.shape)