
import numpy as np
import numpy.linalg as LA
from numpy import zeros
def list2npvector(vect):
    return np.array([float(item) for item in vect])

def string2matrix(vect, m, n):
    res = zeros((m, n))
    cnt = 0
    for (i, j) in np.ndindex(m, n):
        res[i][j] = float(vect[cnt])
        cnt+=1
    return res

def string2list(vect, n):
    return list2npvector(vect)

class W2VModel:

    def __init__(self, filename):
        self.load(filename)

    def load(self, filename):
        print "Loading Word2vec Model : ", filename,'...',
        with open(filename) as f:
            #tmp = f.readline().strip().split()
            #self.n_word = int(tmp[0])
            #self.d_word = int(tmp[1])
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
        #return np.dot(v1, v2)

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

class TWEModel():

    def __init__(self):
        pass

    def find_nearest_word(self, word1, topic1):
        embedding1 = self.generate_embeeding(word1, topic1)
        res = []
        for topic2 in range(self.topic_number):
            for word2 in self.vocab:
                embedding2 = self.generate_embeeding(word2, topic2)
                similarity = self.generate_similarity(embedding1, embedding2)
                if similarity >0.5:
                    res.append((word2, topic2, similarity))

        res = sorted(res, cmp=lambda x, y:-cmp(x[2],y[2]))
        return res

    def generate_similarity(self, embedding1, embedding2):
        norm1 = LA.norm(embedding1)
        norm2 = LA.norm(embedding2)
        return (np.dot(embedding1, embedding2)/(norm1*norm2))

class MVTWEModel(TWEModel):

    def __init__(self, wordfilename, topicfilename, wordmapfile, tassignfilename):
        TWEModel.__init__(self)
        self.load_word(wordfilename)
        self.load_topic(topicfilename)
        self.topicmodel = TopicModel(wordmapfile, tassignfilename)

    def load_word(self, wordfilename):
        self.vocab = {}
        with open(wordfilename) as f:
            for l in f:
                words = l.strip().split()
                word = words[0]
                if word in self.vocab:
                    continue
                else:
                    self.vocab[word] = np.array([float(item) for item in words[1:]])

    def load_topic(self, topicfilename):
        self.topic_number = 0
        self.topic = {}
        with open(topicfilename) as f:
            while True:
                l = f.readline()
                if not l:
                    break
                l = l.strip().split()
                self.topic[self.topic_number] = {}
                self.topic[self.topic_number]['U'] = string2matrix(l, 100, 3)
                l = f.readline().strip().split()
                self.topic[self.topic_number]['V'] = string2matrix(l, 3, 100)
                l = f.readline().strip().split()
                self.topic[self.topic_number]['a'] = string2list(l, 100)
                self.rank = len(self.topic[self.topic_number]['U']) / len(self.topic[self.topic_number]['a'])
                self.topic_number +=1

    def generate_embeeding(self, word, topic):
        if word not in self.vocab:
            return None
        if topic not in self.topic:
            return None
        U = self.topic[topic]['U']
        V = self.topic[topic]['V']
        a = self.topic[topic]['a']
        embedding = self.vocab[word]
        #tmp = np.dot(U, V) +  np.diag(a)
        #return np.dot(tmp, embedding)
        tmp = np.dot(V, embedding)
        tmp = np.dot(U, tmp)
        tmp = tmp + np.dot(np.diag(a), embedding)
        return tmp

    def generate_global_embedding(self, word):
        if word not in self.vocab:
            return None
        return self.vocab[word]

    def find_N_nearest_global_word(self, word, topic):
        embedding1 = self.generate_embeeding(word, topic)
        res = []
        for word2 in self.vocab:
            for topic2 in range(self.topic_number):
                if word2 not in self.topicmodel.model[topic2]:
                    continue
                cnt = self.topicmodel.model[topic2][word2]
                if cnt < 10:
                    continue
                embedding2 = self.generate_embeeding(word2, topic2)
                similarity = self.generate_similarity(embedding1, embedding2)
                if similarity>0.4:
                    res.append((word2, topic2, similarity))

        return sorted(res, cmp=lambda x, y:-cmp(x[2],y[2]))

if __name__=="__main__":
    m = TopicModel("./../ldamodel/wordmap.txt","./../ldamodel/model-final.tassign")
    while True:
        topic = raw_input("Please input the word : ")
        if topic=='EXIT':
            break
        topic = int(topic)
        print "====================================="
        res = m.find_nearest_word(topic)
        for word, cnt in res[:20]:
            print word , cnt
        print "======================================"
    #m = MVTWEModel("./../word.vector","./../topic.vector")
    #while True:
    #    words    = raw_input("Please input the word and the topic : ")
    #    word, topic = words.split(' ')
    #    if word=='EXIT':
    #        break
    #    topic = int(topic)
    #    res = m.find_nearest_word(word, topic)
    #    print "===================================="
    #    for word,topic, similarity in res[:20]:
    #        print word,topic, similarity
    #    print "===================================="

