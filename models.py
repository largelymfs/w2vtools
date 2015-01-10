
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

    def __init__(self, filename):
        W2VModel.__init__(self, filename)

    def load(self, filename):
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


if __name__=="__main__":
    m = TS_MSW2VModel('./Timeseries.conf')
