import getpass
import os
import re
import string as st
import sys
import time
import traceback
import unicodedata
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
# import NLPContractions
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize
from treeinterpreter import treeinterpreter

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def warn_with_traceback(message, category, filename, lineno, file = None, line = None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def get_username():
    return getpass.getuser()


def get_code_dir():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    return code_dir


def print_path(full_path):
    return full_path.replace(get_code_dir(), '$MODELDIR')


def log_writer(my_string):
    my_string = '[%s]: %s.' % (time.strftime('%H:%M:%S'), my_string)
    print(my_string)


def clean_string(string):
    string = string.replace('\n', ' ')
    string = string.rstrip()
    replacements = ['\'', '"']
    for char in replacements:
        string = string.replace(char, '')

    while '  ' in string:
        string = string.replace('  ', ' ')

    return string


def get_feature_importance(model, name, modeldir):
    if not hasattr(model.named_steps['clf'], 'feature_importances_'):
        return

    cols = np.array(model.named_steps['vect'].get_feature_names())
    idx = model.named_steps['genet'].support_
    title = '%s: 20 top features importance' % name
    ax_imp = (pd.Series(model.named_steps['clf'].feature_importances_, index=cols[idx])).nlargest(20).plot(kind='barh', title=title)
    file_name = '%s/output/%s_top_important_features.png' % (modeldir, name)
    plt.savefig(file_name, bbox_inches='tight')
    fig_imp = ax_imp.get_figure()
    fig_imp.clf()


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


def get_sorted_contributed_features(model, data, preds):
    cols = np.array(model.named_steps['vect'].get_feature_names())
    idx = model.named_steps['genet'].support_
    predictors = cols[idx]

    # Get transformed data before applying the classifier
    x_data = data['text']
    for name, transform in model.steps[:-1]:
        x_data = transform.transform(x_data)

    num_rows, num_cols = x_data.shape
    _, _, contributions = treeinterpreter.predict(model.named_steps['clf'], x_data)
    contr_matrix, feature_ordering = np.full(x_data.shape, 0.0), np.full(x_data.shape, 0)
    for i in range(num_rows):
        sorted_contrib = sorted(zip(contributions[i, :, 1], range(num_cols)), key=lambda x: -abs(x[0]))
        contr_matrix[i, :] = list(map(abs, list(zip(*sorted_contrib))[0]))
        feature_ordering[i, :] = list(zip(*sorted_contrib))[1]

    contr_matrix = normalize(contr_matrix, axis=1, norm='l1')
    preds_interpreter = []
    for i in range(num_rows):
        idx_pred = np.argmax(preds)
        pred = ['NOT FAKE', 'FAKE'][(preds[i, 1] >= 0.5)]
        pred_proba = preds[i, idx_pred] * 100
        idx_feature_orders = feature_ordering[i, :10]
        contributed_features = predictors[idx_feature_orders]
        contributed_weights = contr_matrix[i, :10]
        msg_str = 'The news provided most likely is %s with %.1f%% confidence.\nContributed Factors:' % (pred, pred_proba)
        for k in range(10):
            msg_str = '%s\n      %02d. %s: %f' % (msg_str, (k + 1), contributed_features[k], contributed_weights[k])

        preds_interpreter.append(msg_str)

    return preds_interpreter
