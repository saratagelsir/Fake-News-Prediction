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


def _ensure_nltk_resource(resource, download_name=None):
    """
    Minimal, safe NLTK resource check:
    - Avoids failing with LookupError at runtime
    - Avoids aggressive downloads (only attempts if missing)
    """
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        try:
            nltk.download(download_name or resource.split("/")[-1], quiet=True)
            nltk.data.find(resource)
            return True
        except Exception:
            return False


# Keep these but don't force them at import-time unless missing
_ensure_nltk_resource("corpora/stopwords", "stopwords")
_ensure_nltk_resource("corpora/wordnet", "wordnet")
# Punkt can differ across NLTK builds; punkt_tab may be referenced internally in some setups
_ensure_nltk_resource("tokenizers/punkt", "punkt")
_ensure_nltk_resource("tokenizers/punkt_tab/english", "punkt_tab")


def warn_with_traceback(message, category, filename, lineno, file = None, line = None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def get_username():
    return getpass.getuser()


def get_code_dir():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    return code_dir


def print_path(full_path):
    return full_path.replace(get_code_dir(), "$MODELDIR")


def log_writer(my_string):
    my_string = "[%s]: %s." % (time.strftime("%H:%M:%S"), my_string)
    print(my_string)


def clean_string(string):
    string = string.replace("\n", " ")
    string = string.rstrip()
    replacements = ["\'", '"']
    for char in replacements:
        string = string.replace(char, "")

    while "  " in string:
        string = string.replace("  ", " ")

    return string


def get_feature_importance(model, name, modeldir):
    if "clf" not in model.named_steps:
        return
    if not hasattr(model.named_steps["clf"], "feature_importances_"):
        return

    # sklearn >= 1.0 uses get_feature_names_out; older uses get_feature_names
    vect = model.named_steps.get("vect", None)
    if vect is None:
        return

    if hasattr(vect, "get_feature_names_out"):
        cols = np.array(vect.get_feature_names_out())
    else:
        cols = np.array(vect.get_feature_names())

    # If GA feature selection step exists, use its mask; otherwise use all features
    if "genet" in model.named_steps and hasattr(model.named_steps["genet"], "support_"):
        idx = model.named_steps["genet"].support_
        cols_used = cols[idx]
        importances = model.named_steps["clf"].feature_importances_
    else:
        cols_used = cols
        importances = model.named_steps["clf"].feature_importances_

    # Safety: align lengths in case of mismatch
    n = min(len(cols_used), len(importances))
    cols_used = cols_used[:n]
    importances = importances[:n]

    title = "%s: 20 top features importance" % name
    ax_imp = (pd.Series(importances, index=cols_used)).nlargest(20).plot(kind="barh", title=title)
    file_name = "%s/output/%s_top_important_features.png" % (modeldir, name)
    plt.savefig(file_name, bbox_inches="tight")
    fig_imp = ax_imp.get_figure()
    fig_imp.clf()


def cleaning_data(x, new_stopwords):
    # Guard against None / non-string inputs
    if x is None:
        x = ""
    x = str(x)

    # Remove all punctuations from the text
    x = "".join([ch for ch in x if ch not in st.punctuation])

    # Html tag removal BeautifulSoup -
    # BeautifulSoup is a useful library for extracting data from HTML and XML documents
    soup = BeautifulSoup(x, "html.parser")
    x = soup.get_text()

    # Replacing contractions.Contractions is library, solely for expanding contractions
    # x = contractions.fix(x)

    # Remove all the numbers
    x = re.sub(r"[0-9]", "", x)

    # Remove all the special characters
    x = re.sub(r"\W", " ", x)

    # Remove all single characters
    x = re.sub(r"\s+[a-zA-Z]\s+", " ", x)
    x = re.sub(r"\d", "", x)

    # Remove single characters from the start
    x = re.sub(r"^\s*[a-zA-Z]\s+", " ", x)

    # Substituting multiple spaces with single space
    x = re.sub(r"\s+", " ", x, flags=re.I).strip()

    # Removing prefixed "b"
    x = re.sub(r"^b\s+", "", x)

    # Make all letter lowercase
    x = x.lower()

    # Word tokanize and lemmatization for the text
    # Fix punkt_tab errors by falling back to a simple split if tokenizer resources are missing
    try:
        token_text = word_tokenize(x)
    except LookupError:
        token_text = x.split()

    lemmatizer = WordNetLemmatizer()
    x = [lemmatizer.lemmatize(word) for word in token_text]
    x = " ".join(x)

    # Removing stopwords
    try:
        stopwords = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        stopwords = set()

    if new_stopwords:
        stopwords.update(new_stopwords)
    x = " ".join(word for word in x.split() if word not in stopwords)

    # Removing non-ascii words
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")

    return x


def get_sorted_contributed_features(model, data, preds):
    # sklearn >= 1.0 uses get_feature_names_out; older uses get_feature_names
    vect = model.named_steps.get("vect", None)
    if vect is None:
        return ["Could not interpret prediction: missing vectorizer step."]

    if hasattr(vect, "get_feature_names_out"):
        cols = np.array(vect.get_feature_names_out())
    else:
        cols = np.array(vect.get_feature_names())

    # If GA selection exists, use selected predictors; otherwise use all features
    if "genet" in model.named_steps and hasattr(model.named_steps["genet"], "support_"):
        idx = model.named_steps["genet"].support_
        predictors = cols[idx]
    else:
        predictors = cols

    # Get transformed data before applying the classifier
    x_data = data["text"]
    for name, transform in model.steps[:-1]:
        x_data = transform.transform(x_data)

    num_rows, num_cols = x_data.shape

    # treeinterpreter only works for tree-based models; fail gracefully otherwise
    try:
        _, _, contributions = treeinterpreter.predict(model.named_steps["clf"], x_data)
    except Exception:
        # Minimal fallback: report prediction only
        preds_interpreter = []
        for i in range(num_rows):
            if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] >= 2:
                pred = ["NOT FAKE", "FAKE"][(preds[i, 1] >= 0.5)]
                pred_proba = float(np.max(preds[i])) * 100
                msg_str = "The news provided most likely is %s with %.1f%% confidence." % (pred, pred_proba)
            else:
                msg_str = "Prediction generated (feature interpretation not available for this model)."
            preds_interpreter.append(msg_str)
        return preds_interpreter

    contr_matrix, feature_ordering = np.full(x_data.shape, 0.0), np.full(x_data.shape, 0)
    for i in range(num_rows):
        # contributions shape: (n_samples, n_features, n_classes)
        # pick class 1 if exists, else use last class
        class_idx = 1 if contributions.shape[2] > 1 else (contributions.shape[2] - 1)
        sorted_contrib = sorted(zip(contributions[i, :, class_idx], range(num_cols)), key=lambda x: -abs(x[0]))
        contr_matrix[i, :] = list(map(abs, list(zip(*sorted_contrib))[0]))
        feature_ordering[i, :] = list(zip(*sorted_contrib))[1]

    contr_matrix = normalize(contr_matrix, axis=1, norm="l1")
    preds_interpreter = []
    for i in range(num_rows):
        if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] >= 2:
            idx_pred = int(np.argmax(preds[i]))
            pred = ["NOT FAKE", "FAKE"][(preds[i, 1] >= 0.5)]
            pred_proba = float(preds[i, idx_pred]) * 100
        else:
            pred = "FAKE" if float(preds[i]) >= 0.5 else "NOT FAKE"
            pred_proba = float(preds[i]) * 100

        idx_feature_orders = feature_ordering[i, :10]
        # If feature selection is present, mapping may differ; clamp indices safely
        idx_feature_orders = np.clip(idx_feature_orders, 0, len(predictors) - 1)

        contributed_features = predictors[idx_feature_orders]
        contributed_weights = contr_matrix[i, :10]
        msg_str = "The news provided most likely is %s with %.1f%% confidence.\nContributed Factors:" % (pred, pred_proba)
        for k in range(10):
            msg_str = "%s\n      %02d. %s: %f" % (msg_str, (k + 1), contributed_features[k], contributed_weights[k])

        preds_interpreter.append(msg_str)

    return preds_interpreter