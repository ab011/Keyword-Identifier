import nltk
#from nltk. book import *
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
stopwords = nltk.corpus.stopwords.words('english')

#Input Text
file = open("text.txt", 'r')
te = file.read()
file.close()

#Tokenize
tokenize = nltk.word_tokenize(te)

#Stopwords Elimination through stopwords corpus and extra stopwords text files
f = open("FoxStoplist.txt","r")
a = f.read()
a = nltk.word_tokenize(a)
f.close()

f = open("SmartStoplist.txt","r")
b = f.read()
b = nltk.word_tokenize(b)
f.close()

c = a+b
c = list(set(c))
c = c+stopwords
c = list(set(c))
content = [w.lower() for w in tokenize if w.lower() not in c]

#Removing punctuation marks and whitespaces using regular expressions
con = ' '.join(content)

con = re.sub(r'[^\w\d\s]', '', con)
con = re.sub(r'^\s+|\s+?$', '', con)
con = re.sub(r'\s+', ' ', con)


#Catergorizing POS of text using POS tagging
con = con.split()

npt = nltk.pos_tag(con)

#Taking out 
con = [w[0] for w in npt if w[1].startswith('N')]

#Lemmatizing only Nouns present in the text
wordnet = WordNetLemmatizer()
con = [wordnet.lemmatize(w) for w in con]


#
fdist =  nltk.FreqDist(con)
print fdist.items()

#print type(wordlist)

#Book1 is the lemmatized unique words present in text1 of nltk.book corpus
#Book1....n is used to find out the idf for computing tf-idf of the input text
files = ["Book1.txt", "Book2.txt", "Book3.txt", "Book4.txt", "Book5.txt", "Book6.txt", "Book7.txt", "Book8.txt", "Book9.txt"]
book = []
for i in range(len(files)):
    file = open(files[i], 'r')
    text = file.read()
    file.close()
    book.append(text)

doc = [str(book[0]),str(book[1]),str(book[2]),str(book[3]),str(book[4]),
       str(book[5]),str(book[6]),str(book[7]),str(book[8]),str(con)]


#Tfidf is calculated
vectorizer = TfidfVectorizer(ngram_range=(1,1))



X_ngrams = vectorizer.fit(doc)
x_ngrams = vectorizer.transform([doc[9]])


feature_array = np.array(vectorizer.get_feature_names())
tfidf_sorting = np.argsort(x_ngrams.toarray()).flatten()[::-1]

n = 7
top_n = feature_array[tfidf_sorting][:n]
print top_n




