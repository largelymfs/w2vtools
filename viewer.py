import models
from optparse import OptionParser


def show_topicmodel_topic():
    modelfile = "./../mvtwe/Dataset/model-final.tassign"
    wordmapfile = "./../mvtwe/Dataset/wordmap.txt"
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
    modelfile = "./../mvtwe/Dataset/model-final.tassign"
    wordmapfile = "./../mvtwe/Dataset/wordmap.txt"
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
    print "This is TWE"

function_map = {'Topic_topic':show_topicmodel_topic,
                'Topic_word': show_topicmodel_word,
                'TWE' : show_twe}

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
