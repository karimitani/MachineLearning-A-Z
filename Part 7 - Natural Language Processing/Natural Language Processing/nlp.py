import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
re0=sent_tokenize(EXAMPLE_TEXT)  #seperate sentences
print(re0)
"""['Hello Mr. Smith, how are you doing today?', 'The weather is great, and Python is awesome.', 'The sky is pinkish-blue.', "You shouldn't eat cardboard."]"""

new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text) #seperate words

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer() #stem words
for w in words:
    print(ps.stem(w))

"""It
is
import
to
by
veri
pythonli
while
pytho"""

import nltk
nltk.download('state_union')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)
def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)  ##chunking: grouping words together
            chunked = chunkParser.parse(tagged)
            
            print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

            chunked.draw()

    except Exception as e:
        print(str(e))
process_content()

"""[('PRESIDENT', 'NNP'), 
('GEORGE', 'NNP'), ('W.', 'NNP'),, 
   ('THE', 'DT'), ('PRESIDENT', 'NNP'), (':', ':'), ('Thank', 'NNP'),
    ('you', 'PRP'), ('all', 'DT'), ('.', '.')] [('Mr.', 'NNP'), ('Spea
"""
nltk.download('maxent_ne_chunker')
nltk.download('words')
def process_content1():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)   # pull out "entities" like people, places, things, locations, monetary figures
            namedEnt.draw()
    except Exception as e:
        print(str(e))
process_content1()

#Lemitizing
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() #lemmitizing: similar to stem but gives actually word
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("better", pos="a"))

#WordNet
from nltk.corpus import wordnet
syns = wordnet.synsets("program") #WordNet is a lexical database for the English language, part of the NLTK corpus.
print(syns[0].name())
print(syns[0].definition())
print(syns[0].examples())
synonyms = []
antonyms = []
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name()) #finds synonyms and antonyms
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01') #finds similarity between words
print(w1.wup_similarity(w2))

#Text Classification
import nltk
import random
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["stupid"])

word_features = list(all_words.keys())[:3000]
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]  #train the dataset with naive bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

import pickle
save_classifier = open("naivebayes.pickle","wb") #save the algorithm in a pickle file
pickle.dump(classifier, save_classifier)
save_classifier.close()

classifier_f = open("naivebayes.pickle", "rb") #open the saved algorithm to use it
classifier = pickle.load(classifier_f)
classifier_f.close()