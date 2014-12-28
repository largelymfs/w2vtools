import models

if __name__=="__main__":
    model = models.W2VModel("./model")
    while True:
        word = raw_input("Please input a word : ")
        if word=='EXIT':
            break
        res = model.find_N_nearest(word,20)
        print "====================================================="
        for (score, word1) in res:
            print word1, score




