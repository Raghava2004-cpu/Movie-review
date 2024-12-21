import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  re
import string
import nltk
import re
import string
import nltk
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
# Load the data
data = pd.read_csv("C:/Users/ragha/Downloads/IMDB Dataset.csv")
data.head()
data['sentiment'] = data['sentiment'].apply(lambda x : 1 if x == 'positive' else  0)


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

    return text

# Assuming 'data' is your DataFrame with 'review' and 'sentiment' columns
data['review'] = data['review'].apply(clean_text)




x = data['review']
y = data['sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Define a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('mnb', MultinomialNB(alpha =  0.1))
     ])


best_model = pipeline.fit(x_train , y_train)
y_pred = best_model.predict(x_test)
print(accuracy_score(y_test , y_pred))


pickle.dump(best_model ,open('review_prediction.pkl' , 'wb'))
model = pickle.load(open('review_prediction.pkl' ,'rb'))