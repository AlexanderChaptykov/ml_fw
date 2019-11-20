from clickhouse_driver import Client
from gensim.models.wrappers import FastText
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing.text import Tokenizer as keras_tokenizer


class Tokenizer(keras_tokenizer):
    def __init__(self, corpus, num_words):
        super(Tokenizer, self).__init__(num_words)
        super(Tokenizer, self).fit_on_texts(corpus)

    def texts_to_sequences(self, corpus, input_len) -> np.ndarray:
        data = super(type(self), self).texts_to_sequences(corpus)
        data = pad_sequences(data, maxlen=input_len)
        return data


def click_patch(self, req):
    cols = [x[0] for x in self.execute(f'describe ({req})')]
    return pd.DataFrame(self.execute(req), columns=cols)
Client.get_df = click_patch


class Tokenz: 
    def __init__(self, MAX_SIZE_INIT=None, MAX_NB_WORDS=50000, MAX_SEQUENCE_LENGTH=500):
        """get train, target"""
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SIZE_INIT = MAX_SIZE_INIT
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
    
    
    def get_df(self, PATH_TO_CSV):
        """Получаем дф для последующей тренировки"""
        client = Client('localhost')
        train_select = pd.read_csv(PATH_TO_CSV).iloc[:self.MAX_SIZE_INIT]
        text = client.get_df('select url, any(text) as X from url_cleantxt group by url')
        text['url'] = [x[7:] for x in text['url']] 
        df = text.merge(train_select, left_on='url', right_on='domain').drop(['Unnamed: 0', 'domain'], axis=1)
        df['X'] = self.get_first_words(df['X'])
        return df


    def get_first_words(self, corpus):
        """У каждого текста берем только первые 500 слов"""
        new_corp = []
        for txt in corpus:
            new_corp.append(' '.join(txt.split()[:self.MAX_SEQUENCE_LENGTH]))
        return new_corp    


    def train_emb_matrix(self, PATH_TO_MODEL, model_gensim=None):
        print('Загрузка модели')
        if not model_gensim:
            model_gensim = FastText.load_fasttext_format(PATH_TO_MODEL)
        
        print('Создание матрицы')
        embeddings_index = {}
        l = []
        for word in self.tokenizer.word_index:
            try:
                embeddings_index[word] = model_gensim.wv[word] 
            except:
                l.append(word)
        embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, model_gensim.vector_size))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix
        self.model_gensim = model_gensim


    def train_tokenizer(self, corpus):
        self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        self.tokenizer.fit_on_texts(corpus)
        print('tokenizer подготовлен')
        
        
    def transform(self, corpus):
        """Трансформируем дф для дальнейшего обучения"""
        train = self.tokenizer.texts_to_sequences(corpus)
        train = pad_sequences(train, maxlen=self.MAX_SEQUENCE_LENGTH)
        return train



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
            print('cant find' ,word)
    return embedding_matrix

