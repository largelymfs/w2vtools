
import numpy as np
import numpy.linalg as LA

def list2npvector(vect):
    return np.array([float(item) for item in vect])

class W2VModel:

    def __init__(self, filename):
        self.load(filename)

    def load(self, filename):
        print "Loading Word2vec Model : ", filename,'...',
        with open(filename) as f:
            tmp = f.readline().strip().split()
            self.n_word = int(tmp[0])
            self.d_word = int(tmp[1])
            self.content = {}
            for l in f:
                words = l.strip().split()
                word = words[0]
                vector = list2npvector(words[1:])
                self.content[word] = vector
        print "Completed!"

    def similarity(self, worda, wordb):
        if worda not in self.content:
            return 0.0
        if wordb not in self.content:
            return 0.0
        v1 = self.content[worda]
        v2 = self.content[wordb]
        return np.dot(v1, v2)/(LA.norm(v1)*LA.norm(v2))

    def find_N_nearest(self, word, n):
        if word not in self.content:
            return None
        result = [(self.similarity(word, word1), word1) for word1 in self.content.keys() if word1!=word]
        result = sorted(result, cmp=lambda x, y:-cmp(x[0],y[0]))[:n]
        return result

class MSW2VModel(W2VModel):
# the multi-sense word embedding viewer

    def __init__(self, filename):
       #construction function

        W2VModel.__init__(self, filename)

    def load(self, filename):
        #load the MSW2VMODEL

        print "Loading MultiSense Word2vec Model : ", filename,'...',
        with open(filename) as f:
            tmp = f.readline().strip().split()
            self.n_word = int(tmp[0])
            self.d_word = int(tmp[1])
            self.content = {}
            while True:
                l = f.readline()
                if not l:
                    break
                words = l.strip().split()
                word = words[0]
                self.content[word] = {}
                self.content[word]['sense_number'] = int(words[1])
                self.content[word]['prob'] = [int(item) for item in words[2:]]
                total = sum(self.content[word]['prob'])
                self.content[word]['prob'] = [ float(item)/float(total) for item in self.content[word]['prob']]
                self.content[word]['global_embedding'] = np.array([float(item) for item in f.readline().strip().split()])
                self.content[word]['sense_embeddings'] = [ np.array([float(item) for item in f.readline().strip().split()]) for i in range(self.content[word]['sense_number'])]
        print "Completed!"

    def similarity(self, word1, sense1, word2, sense2):
        #if the sense is 0, we use the global embedding
        #return format : the similarity of (word1, sense1) and (word2, sense2)

        if word1 not in self.content:
            return 0.0
        if sense1 == 0:
            embedding1 = self.content[word1]['global_embedding']
        else:
            if sense1 <= self.content[word1]['sense_number']:
                embedding1 = self.content[word1]['sense_embeddings'][sense1 - 1]
            else:
                return 0.0

        if word2 not in self.content:
            return 0.0
        if sense2 == 0:
            embedding2 = self.content[word2]['global_embedding']
        else:
            if sense2 <= self.content[word2]['sense_number']:
                embedding2 = self.content[word2]['sense_embeddings'][sense2 - 1]
            else:
                return 0.0

        return np.dot(embedding1, embedding2)/(LA.norm(embedding1) * LA.norm(embedding2))

    def find_N_nearest(self, word, sense, n):
        #find the nearest global embeddings of the (word, sense)
        #if the sense is 0, means the word's global word embedding
        #return format : a list of n items, each item is (word, similarity)

        if word not in self.content:
            return None
        if sense > self.content[word]['sense_number']:
            return None
        res = [(w, self.similarity(word, sense, w, 0)) for w in self.content if w !=word]
        res = sorted(res, cmp=lambda x, y:-cmp(x[1],y[1]))[:n]
        return res

    def find_N_nearest_whole_word(self, word, n):
        #find the nearest global embeddings of the word under every sense
        #return format : a list of s's lists, each sublist is the n nearest item (word, similarity)

        if word not in self.content:
            return None
        res = [ self.find_N_nearest(word, sense, n) for sense in range(self.content[word]['sense_number'] + 1)]
        return res



class TS_MSW2VModel():

    def __init__(self, conf):
        self.load(conf)

    def load(self, conf):
        self.content = {}
        with open(conf) as fin:
            for l in fin:
                year, filename = l.strip().split()
                year = int(year)
                self.content[year] = MSW2VModel(filename)

class TopicModel():

    def __init__(self, wordmap_fn, tassign_fn):
        self.load_vocab(wordmap_fn)
        self.load_assign(tassign_fn)

    def load_vocab(self, fn):
        self.id2word = {}
        self.word2id = {}
        with open(fn) as fin:
            fin.readline()
            for l in fin:
                word, number = l.strip().split()
                number = int(number)
                self.word2id[word] = number
                self.id2word[number] = word

    def load_assign(self, fn):
        self.model ={}
        with open(fn) as fin:
            for l in fin:
                words = l.strip().split()
                for word in words:
                    word_number , topic_number = word.split(':')
                    word_number = int(word_number)
                    topic_number = int(topic_number)
                    if word_number not in self.id2word:
                        continue
                    word = self.id2word[word_number]
                    if topic_number not in self.model:
                        self.model[topic_number] = {}
                    if word not in self.model[topic_number]:
                        self.model[topic_number][word] = 0
                    self.model[topic_number][word] +=1

    def find_nearest_word(self, topic_number):
        res = self.model[topic_number]
        res = sorted(res.items(), cmp=lambda x, y:-cmp(x[1],y[1]))
        return res

    def find_nearest_topic(self, word):
        res = [(topic_number,self.model[topic_number][word]) for topic_number in self.model if word in self.model[topic_number]]
        res = sorted(res, cmp=lambda x, y : -cmp(x[1],y[1]))
        return res

if __name__=="__main__":
    m = TopicModel("./../ldamodel/wordmap.txt","./../ldamodel/model-final.tassign")
    while True:
        word = raw_input("Please input the word : ")
        if word=='EXIT':
            break
        print "====================================="
        res = m.find_nearest_topic(word)
        for topic_number, cnt in res:
            print topic_number , cnt
        print "======================================"
