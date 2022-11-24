# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 23:19:13 2022

@author: Sayon Conde
"""

# import packages
import streamlit as st
import os
import nltk
nltk.download('omw-1.4')

#nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# TfidfVectorizer Convertir une collection de documents bruts en une matrice de fonctionnalités TF-IDF.
 
# text preprocessing modules
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import joblib
 
import warnings
import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB # classifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    plot_confusion_matrix,)
# text preprocessing modules
# text preprocessing modules
# Download dependency
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
):
    nltk.download(dependency)
    
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)

# load data
data = pd.read_csv("C:\\Users\\tenem\\Downloads\\app1_streamlit\\data\\labeledTrainData.tsv", sep='\t')

# show top five rows of data
data.head()

# check the shape of the data
#data.shape

#We need to check if the dataset has any missing values.
# check missing values in data
data.isnull().sum()

# evalute news sentiment distribution
data.sentiment.value_counts()


stop_words =  stopwords.words('english')
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
        
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)

#clean the review
data["cleaned_review"] = data["review"].apply(text_cleaning)

#split features and target from  data 
X = data["cleaned_review"]
y = data.sentiment.values

#We then split our dataset into train and test data. The test size is 15% of the entire dataset.

# split data into train and validate
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y,
)

# Create a classifier in pipeline
sentiment_classifier = Pipeline(steps=[
                               ('pre_processing',TfidfVectorizer(lowercase=False)),
                                 ('naive_bayes',MultinomialNB())
                                 ])

# train the sentiment classifier 
sentiment_classifier.fit(X_train,y_train)
#We then create a prediction from the validation set.

# test model performance on valid data 
y_preds = sentiment_classifier.predict(X_valid)

accuracy_score(y_valid,y_preds)


#Save Model Pipeline
#The model pipeline will be saved in the model’s directory by using the joblib python package.

#save model 
joblib.dump(sentiment_classifier, "C:\\Users\\tenem\\Downloads\\app1_streamlit\\models\\sentiment_model_pipeline.pkl")

 
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)
 
# load stop words
stop_words = stopwords.words("english")
##Nettoyage du texte


# functon to make prediction
@st.cache
def make_prediction(review):
 
    # clearn the data
    clean_review = text_cleaning(review)
 
    # load the model and make prediction
    model = joblib.load("C:\\Users\\tenem\\Downloads\\app1_streamlit\\models\\sentiment_model_pipeline.pkl")
 
    # make prection
    result = model.predict([clean_review])
 
    # check probabilities
    probas = model.predict_proba([clean_review])
    probability = "{:.2f}".format(float(probas[:, result]))
 
    return result, probability
# Set the app title
st.title("Sentiment Analyisis App")
st.write( "A simple machine laerning app to predict the sentiment of a movie's review")

# Declare a form to receive a movie's review
form = st.form(key="my_form")
review = form.text_input(label="Enter the text of your movie review")
submit = form.form_submit_button(label="Make Prediction")
if submit:
    # make prediction from the input text
    result, probability = make_prediction(review)
 
    # Display results of the NLP task
    st.header("Results")
 
    if int(result) == 1:
        st.write("This is a positive review with a probabiliy of ", probability)
    else:
        st.write("This is a negative review with a probabiliy of ", probability)

#! streamlit run app-streamlit.py()


