import nltk
import string as st
from typing import List, Optional, Callable


def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))


def sent_tokenize(text):
    text = nltk.word_tokenize(text)
    return [x.lower() for x in text]


def remove_small_words(text, thres=2):
    return [x for x in text if len(x) > thres]


def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]


def stemming(text):
    ps = nltk.PorterStemmer()
    return [ps.stem(word) for word in text]


def lemmatize(text):
    word_net = nltk.WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]


def preprocess_text(input_text: str, processing_function_list: Optional[List[Callable]] = None):
    if processing_function_list is None:
        processing_function_list = [remove_punct,
                                    sent_tokenize,
                                    remove_small_words,
                                    remove_stopwords,
                                    stemming,
                                    lemmatize]
    for func in processing_function_list:
        input_text = func(input_text)

    return input_text