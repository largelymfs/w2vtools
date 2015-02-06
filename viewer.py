import models
from optparse import OptionParser


def show_topicmodel_topic():
    modelfile = "./../Dataset/model-final.tassign"
    wordmapfile = "./../Dataset/wordmap.txt"
    m = models.TopicModel(wordmapfile, modelfile)
    while True:
        number = raw_input("Please Input the number : ")
        try:
            number = int(number)
        except:
            print "ERROR"
            continue
        if number==-1:
            break
        res = m.find_nearest_word(number)
        res = res[:30]
        print "==================================="
        for word, cnt in res:
            print word, cnt
        print "==================================="

def show_topicmodel_word():
    modelfile = "./../Dataset/model-final.tassign"
    wordmapfile = "./../Dataset/wordmap.txt"
    m = models.TopicModel(wordmapfile, modelfile)
    while True:
        word = raw_input("Please Input the word : ")
        if word=='EXIT':
            break
        res = m.find_nearest_topic(word)[:30]
        print "===================================="
        for topic, cnt in res:
            print topic, cnt
        print "===================================="

def show_twe():
    wordfile = "./../new_gensim/word.vector.wik"
    topicfile = "./../new_gensim/topic.vector.wik"
    wordmapfile = "./../Dataset/wordmap.txt"
    tassignfilename = "./../Dataset/model-final.tassign"
    m = models.MVTWEModel(wordfile, topicfile, wordmapfile, tassignfilename)
    while True:
        ress = raw_input("Please enter the word and topic :  ")
        ress = ress.strip().split()
        word = ress[0]
        topic = int(ress[1])
        if topic==-1:
            break
        if word=='EXIT':
            break
        print "============================================"
        res = m.find_N_nearest_global_word(word, topic)
        for word, topic, similarity in res[:30]:
            print word, topic, similarity
        print "============================================"

def show_word2vec():
    modelfilename = "./../new_gensim/word.vector.wik"
    m = models.W2VModel(modelfilename)
    while True:
        word = raw_input("Please enter the word : ")
        if word=='EXIT':
            break
        res = m.find_N_nearest(word, 30)
        print "=========================================="
        for word, similarity in res[:100]:
            print word, similarity
        print "==========================================="


function_map = {'Topic_topic':show_topicmodel_topic,
                'Topic_word': show_topicmodel_word,
                'TWE' : show_twe,
                'word2vec' : show_word2vec}

if __name__=='__main__':

    #initialize the optparse
    parser = OptionParser()
    parser.add_option("-m","--model",dest ='model',help='Please enter the model name: Topic_word,\
                      Topic_topic, TWE')
    (options, args) = parser.parse_args()
    model = options.model

    if model not in function_map:
        print "Error, Please enter the correct model name"
    else:
        func = function_map[model]
        func()
