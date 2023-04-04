import spacy
import numpy as np
nlp = spacy.load("en_core_web_lg")

def tokenizeWithLemma(sentence):
    doc = nlp(sentence)
    words = []   
    for token in doc:
        #it will be better to remove also the stop_words , but data is too small ,so i decided to keep it :)
        if (not token.is_punct) :
            word = token.lemma_
            words.append(word.lower())
    return words

def bagOfWords(arrayOfWords , wordsDic):
    sentArray = np.zeros(len(wordsDic)+1)
    for word in arrayOfWords:
        if word in wordsDic:
            sentArray[wordsDic[word]] += 1
            continue
        sentArray[-1] += 1 
    return sentArray
