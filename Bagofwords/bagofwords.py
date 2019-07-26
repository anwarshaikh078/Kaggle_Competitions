import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

#loading the training file
train = pd.read_csv('dataset\labeledTrainData.tsv',header=0, \
                    delimiter="\t", quoting=3)

print(train.shape)
print(train.columns.values)

import re
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

#helper function to clean the train data
def review_to_words(raw_review):
    label = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-bA-B]"," ",label)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]

    return(" ".join(meaningful_words))


num_reviews = train["review"].size

clean_train_reviews = []
#cleaning the train data
for i in range(0,num_reviews):
    if ((i + 1) % 1000 == 0):
        print ("train Review %d of %d\n" % ( i+1, num_reviews))

    clean_train_reviews.append(review_to_words(train["review"][i]))

from sklearn.feature_extraction.text import CountVectorizer
#for vectoriztion process
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

train_clean_review = vectorizer.fit_transform(clean_train_reviews)

train_clean_review = train_clean_review.toarray()

#random forest model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_clean_review, train["sentiment"])

#for submission file
test = pd.read_csv('dataset\checkfile.tsv', header=0, delimiter="\t", \
                   quoting=3 )

#cleaning the test data
num_test_reviews = test["review"].size
clean_test_review = []
for i in range(0,num_test_reviews):
    if ((i + 1) % 1000 == 0):
        print ("test Review %d of %d\n" % ( i+1, num_test_reviews ))
    clean_test_review.append(review_to_words(test["review"][i]))

train_clean_review = vectorizer.transform(clean_test_review)
train_clean_review = train_clean_review.toarray()

#predicting output
result = forest.predict(train_clean_review)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )


