import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def is_valid_txt(txt: str) -> bool:
    """get text return where True good text"""
    if type(txt) != str:
        return False
    if len(txt.split()) == 0 or pd.isnull(txt):
        return False
    return True


def get_data_token(corpus, input_len, max_features=95000):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(corpus)
    train = tokenizer.texts_to_sequences(corpus)
    train = pad_sequences(train, maxlen=input_len)
    return train, tokenizer
