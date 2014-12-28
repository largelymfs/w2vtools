import scipy as sci
import scipy.stats as ST
import models

class Tester:
    def __init__(self, standard_file):
        self.word_pair= []
        self.std = []
        with open(standard_file) as f:
            for l in f:
                worda, wordb, score = l.strip().split()
                score = float(score)
                self.word_pair.append((worda, wordb))
                self.std.append(score)

    def test(self, model):
        result = [model.similarity(worda, wordb) for (worda, wordb) in self.word_pair]
        return ST.spearmanr(result, self.std)

if __name__=="__main__":
    input_model = "./../word2vec/model"
    input_st = "./wordsim353/wordsim_relatedness_goldstandard.txt"
    model = models.W2VModel(input_model)
    t = Tester(input_st)
    result = t.test(model)
    print "relationship spearman correlation : ", result[0]*100,result[1]
    input_st2 = "./wordsim353/wordsim_similarity_goldstandard.txt"
    t1 = Tester(input_st2)
    result = t1.test(model)
    print "similarity spearman correlation : ", result[0]*100, result[1]
