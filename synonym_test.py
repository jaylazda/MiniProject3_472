from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim import similarities
from gensim import downloader
import pandas as pd
import random

def analysis(model, num_vec, vocab_size, num_correct, num_no_guesses):
    file = open('analysis.csv','a')
    file.write(f'{model}-{num_vec},{vocab_size},{num_correct},{num_no_guesses},{num_correct/num_no_guesses}\n')
    file.close()

corpus = pd.read_csv('synonyms.csv')
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
# file = open("word2vec-google-news-300-details.csv","w")
# model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
# file = open("wiki-news-300d-1M-details.csv","w")
# model = downloader.load('glove-wiki-gigaword-300')
# file = open("glove-wiki-gigaword-300-details.csv","w")
# model = downloader.load('glove-twitter-25')
# file = open("glove-twitter-25-details.csv","w")
# model = downloader.load('glove-twitter-100')
# file = open("glove-twitter-100-details.csv","w")

num_correct = 0
num_no_guesses = 0

for i in range (len(corpus)):
    label = ""
    flags = [False, False, False, False]
    question = corpus.loc[i,'question']
    answer = corpus.loc[i,'answer']
    guess = ""
    try:
        dist0 = model.similarity(question, corpus.loc[i,'0'])
    except:
        flags[0] = True
    try:
        dist1 = model.similarity(question, corpus.loc[i,'1'])
    except:
        flags[1] = True
    try:
        dist2 = model.similarity(question, corpus.loc[i,'2'])
    except:
        flags[2] = True
    try:
        dist3 = model.similarity(question, corpus.loc[i,'3'])
    except:
        flags[3] = True
    
    if all(flags):
        result = "guess"
        guess = corpus.loc[i,f'{random.randrange(4)}']
    else:
        closest = max(dist0, dist1, dist2, dist3)
        if closest == dist0:
            guess = corpus.loc[i,'0']
        elif closest == dist1:
            guess = corpus.loc[i,'1']
        elif closest == dist2:
            guess = corpus.loc[i,'2']
        elif closest == dist3:
            guess = corpus.loc[i,'3']
        result = "wrong"
        if (answer == guess):
            result = "correct"
            num_correct += 1
        num_no_guesses += 1
    file.write(question+','+answer+','+guess+','+result+'\n')

file.close()
# analysis('word2vec-google-news', 300, len(model), num_correct, num_no_guesses)
# analysis('wiki-news', 300, len(model), num_correct, num_no_guesses)
# analysis('glove-wiki-gigaword', 300, len(model), num_correct, num_no_guesses)
# analysis('glove-twitter', 25, len(model), num_correct, num_no_guesses)
# analysis('glove-twitter', 100, len(model), num_correct, num_no_guesses)

# Reference for wiki-news word embeddings
# @inproceedings{mikolov2018advances,
#   title={Advances in Pre-Training Distributed Word Representations},
#   author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
#   booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
#   year={2018}
# }