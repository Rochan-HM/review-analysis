import multiprocessing
import spacy
import re
import nltk

import pandas as pd
import numpy as np

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


nltk.download("stopwords")
nltk.download("punkt")

sw = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

# Header for the application.
# Make a heading with the logo from https://aialoe.org/wp-content/uploads/2023/01/dark-text-no-tag-cener.jpg
# and the title "CARES"
# Everything should be centered
HEADER = """
<div style="text-align: center;">
<img src="https://aialoe.org/wp-content/uploads/2023/01/dark-text-no-tag-cener.jpg" alt="Logo" width="300">
<h1>CARES</h1>
<h2>Classroom Assessment Review and Evaluation System</h2>
</div>
""".strip()

EMBEDDING_MODEL_MSG = """
`doc2vec` is a neural network-based model for representing documents as fixed-length vectors. It is trained on a dataset of text documents and can be used for tasks such as document similarity comparison and document classification.

The `universal-sentence-encoder` and its variants (`universal-sentence-encoder-large`, `universal-sentence-encoder-multilingual`, `universal-sentence-encoder-multilingual-large`) are models that generate embeddings for individual sentences or short paragraphs. These embeddings can be used for tasks such as semantic similarity comparison, text classification, and machine translation. The main difference between these models is their size and the language they are trained on. For example, `universal-sentence-encoder-multilingual` is trained on a diverse set of languages, while `universal-sentence-encoder-large` is a larger version of the model with more parameters.

`distiluse-base-multilingual-cased` is a smaller, faster version of the `universal-sentence-encoder-multilingual` model. This distill version has similar performance with the original version with less computation required.

`all-MiniLM-L6-v2` and `paraphrase-multilingual-MiniLM-L12-v2` are both variants of MiniLM models, which are small language models that can be fine-tuned for a wide range of natural language understanding tasks. The main difference between these two models is the size of the model and the tasks they are trained on. The `all-MiniLM-L6-v2` is a smaller model, while the `paraphrase-multilingual-MiniLM-L12-v2` is larger, and specifically trained on paraphrase identification tasks.
""".strip()

SPEED_MSG = """
This parameter is only used when using `doc2vec` as embedding_model.

It will determine how fast the model takes to train. The fast-learn option is the fastest and will generate the lowest quality vectors. The learn option will learn better quality vectors but take a longer time to train. The deep-learn option will learn the best quality vectors but will take significant time to train. 
""".strip()


def get_num_cpu_cores():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()


def read_data_csv(path):
    """Read data from a CSV file."""
    try:
        return pd.read_csv(path, engine="python")
    except:
        raise Exception(
            "Could not read CSV file. Please check the encoding of the file."
        )


# Preprocess the data
def _remove_hyperlinks(text):
    """Remove hyperlinks and markup"""
    result = re.sub("<[a][^>]*>(.+?)</[a]>", "Link.", text)
    result = re.sub("&gt;", "", result)
    result = re.sub("&#x27;", "'", result)
    result = re.sub("&quot;", '"', result)
    result = re.sub("&#x2F;", " ", result)
    result = re.sub("<p>", " ", result)
    result = re.sub("</i>", "", result)
    result = re.sub("&#62;", "", result)
    result = re.sub("<i>", " ", result)
    result = re.sub("\n", "", result)
    return result


def _remove_numbers(text):
    """Remove numbers"""
    result = re.sub(r"\d+", "", text)
    return result


def _remove_punctuation(text):
    """Remove punctuation"""
    result = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', ","))
    return result


def _remove_whitespace(text):
    """Remove extra whitespace"""
    result = re.sub(" +", " ", text)
    return result


def _remove_stopwords(text):
    """Remove stopwords"""
    result = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(result)


def _stem_words(text):
    """Stem words"""
    ss = SnowballStemmer("english")
    word_tok = nltk.word_tokenize(text)
    stemmed_words = [ss.stem(word) for word in word_tok]
    return " ".join(stemmed_words)


def _apply_preprocessing(df, column, new_column):
    """Apply preprocessing to a dataframe"""

    df[new_column] = df[column].apply(_remove_hyperlinks)
    df[new_column] = df[column].apply(_remove_numbers)
    df[new_column] = df[column].apply(_remove_punctuation)
    df[new_column] = df[column].apply(_remove_whitespace)
    df[new_column] = df[column].apply(_remove_stopwords)
    df[new_column] = df[column].apply(_stem_words)

    return df


def _compute_IDF(documents):
    """
    Compute the IDF for each word in the corpus.
    """

    res = []

    for doc in documents:
        doc = nlp(doc)
        temp_sentence = ""
        for token in doc:
            if token.pos_ in ["VERB", "NOUN", "ADJ"] or (token.dep_ == "dobj"):
                temp_sentence += token.lemma_.lower() + " "
        res.append(temp_sentence)

    count = Counter()
    for doc in res:
        count.update(set(doc.split()))

    total = np.sum(list(count.values()))
    idf = {word: round(np.log2(total / ct)) for word, ct in count.items()}
    return idf


def _get_suggested_titles(documents):
    """
    Get suggested titles for each cluster.
    """
    cluster_df = pd.DataFrame(documents, columns=["Text"])
    cluster_df = _apply_preprocessing(cluster_df, "Text", "Cleaned")

    idf = _compute_IDF(cluster_df["Cleaned"])
    label = _extract_cluster_label(cluster_df["Cleaned"], idf)

    return label


def _most_common(documents, n_words, idf):
    """
    Get the most common words in the corpus.
    """
    doc_count = Counter(documents)
    for w in list(doc_count):
        if doc_count[w] == 1:
            pass
        else:
            doc_count[w] = doc_count[w] * idf.get(w, 1)

    return doc_count.most_common(n_words)


def _extract_cluster_label(clustered_docs, idf):
    """
    Extract a label for the cluster based on most common verbs, objects, and nouns.
    """
    nlp = spacy.load("en_core_web_sm")

    verbs = []
    objects = []
    nouns = []
    adjectives = []

    for doc in nlp.pipe(clustered_docs):
        for tok in doc:
            if tok.is_stop or not str(tok).strip():
                continue
            if tok.dep_ == "VERB":
                verbs.append(tok.lemma_.lower())
            elif tok.dep_ == "dobj":
                objects.append(tok.lemma_.lower())
            elif tok.pos_ == "NOUN":
                nouns.append(tok.lemma_.lower())
            elif tok.pos_ == "ADJ":
                adjectives.append(tok.lemma_.lower())

    #  Get most common verbs, objects, nouns, and adjectives
    if verbs:
        verb = _most_common(verbs, 1, idf)[0][0]
    else:
        verb = ""

    if objects:
        obj = _most_common(objects, 1, idf)[0][0]
    else:
        obj = ""

    if nouns:
        noun = _most_common(nouns, 1, idf)[0][0]
    else:
        noun = ""

    if len(set(nouns)) > 1:
        noun2 = _most_common(nouns, 2, idf)[1][0]

    if adjectives:
        adj = _most_common(adjectives, 1, idf)[0][0]
    else:
        adj = ""

    # concatenate the most common verb-object-noun1
    label_words = [verb, obj]

    for word in [noun, noun2]:
        if word not in label_words:
            label_words.append(word)

    if "" in label_words:
        label_words.remove("")

    label = "_".join(label_words)

    return label


def extract_labels(model):
    """
    Extract a label for the cluster based on most common verbs, objects, and nouns.
    """

    num_topics = model.get_num_topics()
    num_docs = model.get_topic_sizes()[0]

    labels = []

    for i in range(num_topics):
        topic_docs = model.search_documents_by_topic(i, num_docs=num_docs[i])[0]
        label = _get_suggested_titles(topic_docs)
        labels.append(label)

    return labels
