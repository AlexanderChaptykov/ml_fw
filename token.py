import pandas as pd
import numpy as np
import re
import logging
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.text import Tokenizer as keras_tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from nltk.stem import WordNetLemmatizer
from selectolax.parser import HTMLParser
from nltk.corpus import stopwords
from pymystem3 import Mystem
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()
stem = Mystem()

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'

logging.basicConfig(format=FORMAT)
logger = logging.getLogger()


class Funct:
    """
    How to get a function name as a string?
    I like using a function decorator. I added a class,
    which also times the function time.
    Assume gLog is a standard python logger:

    how to use
    @Funct.log
    def my_func():
        pass

    """
    def __init__(self, funcName):
        self.funcName = funcName

    def __enter__(self):
        logger.info(f'{self.funcName} started')
        self.init_time = datetime.datetime.now()
        return self

    def __exit__(self, type, value, tb):
        logger.info(f'{self.funcName} finished: {datetime.datetime.now() - self.init_time}')


    @classmethod
    def log(cls, func):
        def func_wrapper(*args, **kwargs):
            with cls(func.__name__):
                return func(*args, **kwargs)

        return func_wrapper



class Tokenizer(keras_tokenizer):
    def __init__(self, corpus, num_words):
        super(Tokenizer, self).__init__(num_words)
        super(Tokenizer, self).fit_on_texts(corpus)

    def texts_to_sequences(self, corpus, input_len, padding='post') -> np.ndarray:
        data = super(type(self), self).texts_to_sequences(corpus)
        data = pad_sequences(data, maxlen=input_len, padding=padding)
        return data

stop_words = set(stopwords.words('english'))

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


class Preparation:
    """
    # example:
    params = {
    "html_to_text": True,
    "chars_n_digs_only": True,
    "stop_words": True,
    "lemm": True,
    'cut_first_words': False
    }
    a = Preparation(corpus, params).texts
     # get prepared texts
    """

    def __init__(self, corpus, params):
        self.params = params
        self.corpus = corpus
        self.start()
        # self.__dict__.update(params)
        # 2 var
        # for func in params:
        #    if params[func]:
        #        getattr(self, func)()


    def start(self):
        self.texts = []
        #print(self.params)
        for text in self.corpus:
            text = str(text)
            # print(text)
            for func in self.params:
                if self.params[func]:
                    text = getattr(self, func)(text, *[self.params[func]])
                #print(text)
            self.texts.append(text)


    def html_to_text(self, html, *args):
        raw = BeautifulSoup(html, 'html.parser').text
        raw = ' '.join(raw.split())
        return raw

        tree = HTMLParser(html)
        if tree.body is None:
            return 'return None'
        for tag in tree.css('script'):
            tag.decompose()
        for tag in tree.css('style'):
            tag.decompose()
        text = tree.body.text(separator='\n')
        text = ' '.join(text.split())
        return text


    def text_cleaner(self, text, clean_stopwords=False, remove_short_words=False):
        import inspect

        this_function_name = inspect.currentframe().f_code.co_name

        """We will perform the below preprocessing tasks for our data:

            1.Convert everything to lowercase

            2.Remove HTML tags

            3.Contraction mapping

            4.Remove (‘s)

            5.Remove any text inside the parenthesis ( )

            6.Eliminate punctuations and special characters

            7.Remove stopwords

            8.Remove short words

            Let’s define the function:"""

        newString = text.lower()
        newString = BeautifulSoup(newString, "lxml").text
        newString = re.sub(r'\([^)]*\)', '', newString)
        newString = re.sub('"', '', newString)
        newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
        newString = re.sub(r"'s\b", "", newString)
        newString = re.sub("[^a-zA-Z]", " ", newString)
        newString = re.sub('[m]{2,}', 'mm', newString)

        if clean_stopwords:
            tokens = [w for w in newString.split() if not w in stop_words]
        else:
            tokens = newString.split()

        if remove_short_words:
            tokens = [x for x in tokens if len(x) > 1]

        return (" ".join(tokens)).strip()


    def lemm(self, text, *args):
        stem = Mystem()
        res = stem.lemmatize(text)
        return ''.join(res[:-1])


    def stop_words(self, text, *args):
        text = text.split()
        russian_stopwords = stopwords.words("russian")
        stopwords_extension = ['ко']
        russian_stopwords = russian_stopwords + stopwords_extension
        stops = set(stopwords.words("english")) | set(russian_stopwords)
        res = ' '.join([w for w in text if not ((w in stops) or len(w) == 1)])
        return res


    def chars_n_digs_only(self, text, *args):
        # print(text)
        # заменяю переносы строк, табуляции и технические символы

        # оставляю только слова и перевожу в нижний регистр
        text = re.sub(r'[^a-zA-Zа-яА-Я ]+', ' ', str(text)).lower()
        text = ' '.join(str(text).split())
        return text


    def cut_first_words(self, text, *args):
        # print('--')
        size = args[0]
        return ' '.join(text.split()[:size])


class Modeling:
    def __init__(self):
        pass
    
    def valid(self, train, target, model, model_conf):
        skf = StratifiedKFold(n_splits=4, random_state=1001)
        starttime = timer(None)
        dfs = []
        for i, (train_index, test_index) in enumerate(skf.split(train, target.argmax(1))):
            #start_time = timer(None)
            X_train, X_val = train[train_index], train[test_index]
            y_train, y_val = target[train_index], target[test_index]

            """            
            #train_ids, val_ids = tr_ids[train_index], tr_ids[test_index]
            
            # This is where we define and compile the model. These parameters are not optimal, as they were chosen 
            # to get a notebook to complete in 60 minutes. Other than leaving BatchNormalization and last sigmoid 
            # activation alone, virtually everything else can be optimized: number of neurons, types of initializers, 
            # activation functions, dropout values. The same goes for the optimizer at the end.

            #########
            # Never move this model definition to the beginning of the file or anywhere else outside of this loop. 
            # The model needs to be initialized anew every time you run a different fold. If not, it will continue 
            # the training from a previous model, and that is not what you want.
            #########

            # This definition must be within the for loop or else it will continue training previous model


            # This is where we repeat the runs for each fold. If you choose runs=1 above, it will run a 
            # regular N-fold procedure.

            #########
            # It is important to leave the call to random seed here, so each run starts with a different seed.
            #########
            """
            for run in range(runs):
                print('\n Fold %d - Run %d\n' % ((i + 1), (run + 1)))
                np.random.seed()

                # Lots to unpack here.

                # The first callback prints out roc_auc and gini values at the end of each epoch. It must be listed 
                # before the EarlyStopping callback, which monitors gini values saved in the previous callback. Make 
                # sure to set the mode to "max" because the default value ("auto") will not handle gini properly 
                # (it will act as if the model is not improving even when roc/gini go up).

                # CSVLogger creates a record of all iterations. Not really needed but it doesn't hurt to have it.

                # ModelCheckpoint saves a model each time gini improves. Its mode also must be set to "max" for reasons 
                # explained above.

                callbacks = [
                    roc_auc_callback(training_data=(X_train, y_train),validation_data=(X_val, y_val)),  # call this before EarlyStopping
                    EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),
                    CSVLogger('keras-5fold-run-01-v1-epochs.log', separator=',', append=False),
                    ModelCheckpoint(
                            'keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check',
                            monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                            save_best_only=True,
                            verbose=1)
                ]

                # The classifier is defined here. Epochs should be be set to a very large number (not 3 like below) which 
                # will never be reached anyway because of early stopping. I usually put 5000 there. Because why not.

                nnet = KerasClassifier(
                    build_fn=model,
                    conf=model_conf,
                    # Epoch needs to be set to a very large number ; early stopping will prevent it from reaching
                    #            epochs=5000,
                    epochs=10,
                    batch_size=250,
                    validation_data=(X_val, y_val),
                    verbose=2,
                    shuffle=True,
                    callbacks=callbacks)

                fit = nnet.fit(X_train, y_train)
                
                # We want the best saved model - not the last one where the training stopped. So we delete the old 
                # model instance and load the model from the last saved checkpoint. Next we predict values both for 
                # validation and test data, and create a summary of parameters for each run.

                del nnet
                nnet = load_model('keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
                
                dfs += [get_results(
                    nnet, 
                    [[X_train, y_train], [X_val, y_val], [test, test_y]], 
                    f"f{i}_r{run}")
                    ]

def get_embeddings(emb_model, embed_size, tokenizer):
    embeddings_index = {}
    l = []
    for word in tokenizer.word_index:
        try:
            embeddings_index[word] = emb_model[word]
        except:
            l.append(word)
    embedding_matrix = np.random.uniform(
        low=-0.05,
        high=0.05,
        size=(len(tokenizer.word_index) + 1, embed_size))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            print('cant find', word)
    return embedding_matrix

