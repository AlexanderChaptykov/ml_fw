import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from clickhouse_driver import Client


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


def click_patch(self, req):
    cols = [x[0] for x in self.execute(f'describe ({req})')]
    return pd.DataFrame(self.execute(req), columns=cols)


def obj_dic(d):
    """Convert dict to object"""
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, obj_dic(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


Client.get_df = click_patch
