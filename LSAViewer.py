import sys

def load_stoplist(fn):
    with open(fn) as f:
        s = [l.strip() for l in f]
    return s

class LSAModel():

    def __init__(self, filename, window_size):
        self.window_size = window_size
        self.load(filename)

    def load(self, filename):
        self.data = {}
        with open(filename) as f:
            for l in f:
                words = l.strip().split()
                l = len(words)
                for i in range(l):
                    if words[i] not in self.data:
                        self.data[words[i]] = {}
                    start = i - self.window_size
                    finish = i + self.window_size
                    start = max(0, start)
                    finish = min(l - 1, finish)
                    for k in range(start, finish + 1):
                        if words[k] not in self.data[words[i]]:
                            self.data[words[i]][words[k]] = 0
                        self.data[words[i]][words[k]] +=1

    def get_frequent_words(self, word):
        res = self.data[word].items()
        return sorted(res, cmp=lambda x, y:-cmp(x[1],y[1]))

class TopicalLSAModel():

    def __init__(self, wordmapfn, tassignfn, window_size):
        self.window_size = window_size
        print "Loading the wordmap",
        sys.stdout.flush()
        self.load_wordmap(wordmapfn)
        print "ok"
        print "Loading the model",
        sys.stdout.flush()
        self.load_tassign(tassignfn)
        print "ok"

    def load_wordmap(self, filename):
        self.id2word = {}
        with open(filename) as f:
            f.readline()
            for l in f:
                word, number = l.strip().split()
                number = int(number)
                self.id2word[number] = word

    def load_tassign(self, filename):
        self.data = {}
        with open(filename) as f:
            for l in f:
                words = l.strip().split()
                words = [item.split(':') for item in words]
                words = [(int(item1),int(item2)) for (item1, item2) in words]
                words = [(self.id2word[word_id], topic_id) for (word_id, topic_id) in words if word_id in self.id2word]
                l = len(words)
                for (i, (word, topic_id)) in enumerate(words):
                    if (word, topic_id) not in self.data:
                        self.data[(word, topic_id)] = {}
                    st = max(0, i - self.window_size )
                    fi = min(l -1, i + self.window_size)
                    for (word0, topic_id0) in words[st:fi+1]:
                        #if (word0, topic_id0) not in self.data[(word, topic_id)]:
                        #    self.data[(word, topic_id)][(word0, topic_id0)] = 0
                        #self.data[(word, topic_id)][(word0, topic_id0)] += 1
                        if word0 not in self.data[(word, topic_id)]:
                            self.data[(word, topic_id)][word0] = 0
                        self.data[(word, topic_id)][word0] +=1
    def get_frequent_wtpair(self, word, topic):
        res = self.data[(word, topic)].items()
        return (sorted(res, cmp=lambda x, y : -cmp(x[1],y[1])), self.data[(word, topic)][word])

    def get_frequent(self, word):
        keys = [key for key in self.data if key[0]==word]
        return [(key, self.get_frequent_wtpair(key[0],key[1])) for key in keys]

if __name__=="__main__":
    stoplist = load_stoplist("./stoplist.txt")

    #m = TopicalLSAModel("./../Dataset/wordmap.txt","./../Dataset/model-final.tassign", 10)
    #m = TopicalLSAModel("./../Dataset/wordmap.txt","./test.model", 5)
    m = TopicalLSAModel("./../multi_fastLDA/src/wordmap.txt","./../multi_fastLDA/src/model-final.tassign", 10)
    while True:
        word = raw_input("Please Input A Word : ")
        if word=='EXIT':
            break
        result = m.get_frequent(word)
        result = sorted(result, cmp=lambda x,y:-cmp(x[1],y[1]))[:5]
        for ((wordtem, topictem), (res,_))  in result:
            res = [(word, cnt) for word, cnt in res if word not in stoplist]
            res = res[:10]
            print wordtem, topictem, " : ",
            for item in res:
                print  item[0], item[1],
            print
