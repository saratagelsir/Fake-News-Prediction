import os
import re
import string as st

import gradio as gr
import joblib
import nltk
import unicodedata
# import NLPContractions
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def cleaning_data(x, new_stopwords):
    # Remove all punctuations from the text
    x = "".join([ch for ch in x if ch not in st.punctuation])

    # Html tag removal BeautifulSoup -
    # BeautifulSoup is a useful library for extracting data from HTML and XML documents
    soup = BeautifulSoup(x, "html.parser")
    x = soup.get_text()

    # Replacing contractions.Contractions is library, solely for expanding contractions
    # x = contractions.fix(x)

    # Remove all the numbers
    x = re.sub(r'[0-9]', '', str(x))

    # Remove all the special characters
    x = re.sub(r'\W', ' ', x)

    # Remove all single characters
    x = re.sub(r'\s+[a-zA-Z]\s+', ' ', x)
    x = re.sub(r'\d', '', x)

    # Remove single characters from the start
    x = re.sub(r'\^[a-zA-Z]\s+', ' ', x)

    # Substituting multiple spaces with single space
    x = re.sub(r'\s+', ' ', x, flags=re.I)

    # Removing prefixed 'b'
    x = re.sub(r'^b\s+', '', x)

    # Make all letter lowercase
    x = str(x).lower()

    # Word tokanize and lemmatization for the text
    token_text = word_tokenize(x)
    lemmatizer = WordNetLemmatizer()
    x = [lemmatizer.lemmatize(word) for word in token_text]
    x = ' '.join(x)

    # Removing stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(new_stopwords)
    x = ' '.join(word for word in x.split() if word not in stopwords)

    # Removing non-ascii words
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')

    return x


def main_caller(news, news_type):
    # Define the caller json string
    trained_model_file = os.path.join('FakeNewsPrediction', 'data', news_type.lower().replace(' ', '_') + '_trained_model.pkl')
    model = joblib.load(trained_model_file)
    news = cleaning_data(news, [])
    model_pred = model.predict_proba([news])[0][1]
    model_output = 'The news provided most likely is %s [%.0f%%]' % (['NOT FAKE', 'FAKE'][int(model_pred > 0.5)], model_pred * 100)
    return model_output


info = 'Machine learning based app for fake news detection'
iface = gr.Interface(
    fn=main_caller,
    inputs=[gr.inputs.Textbox(lines=20, placeholder="News Text or URL Here..."), gr.inputs.Radio(['General News', 'COVID19 News'])],
    outputs='text',
    title='Fake News Prediction',
    description=info)
iface.launch()
