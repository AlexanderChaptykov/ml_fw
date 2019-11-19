import gc
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Fw_ml:
    def prepare_data(self, df, embed_path_or_model, emb_size):
        self.text_to_data(df)
        self.get_embeddings(embed_path_or_model, emb_size)


    def text_to_data(self, df, md = {"txt":"text", "label": "label"}):
        print("text_to_data")
        max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
        self.input_shape = 70 # max number of words in a question to use

        ## fill up the missing values
        #data = train_df["NormText"].fillna("_##_").values
        self.data = df[md['txt']].values
        self.label = df[md['label']].values
        ## Tokenize the sentences
        self.tokenizer = Tokenizer(num_words=max_features)
        print("tokenizer.fit_on_texts")
        self.tokenizer.fit_on_texts(self.data)
        self.data = self.tokenizer.texts_to_sequences(self.data)
        self.data = pad_sequences(self.data, maxlen=self.input_shape)
        self.target_shape = 1


    def get_embeddings(self, emb_model, emb_size):
        embeddings_index = {}
        l = []
        for word in self.tokenizer.word_index:
            try:
                embeddings_index[word] = emb_model[word] 
            except:
                l.append(word)

        self.embeddings = np.random.uniform(
            low=-0.05, 
            high=0.05, 
            size=(len(self.tokenizer.word_index) + 1, emb_size))

        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embeddings[i] = embedding_vector
            else:
                print(word)


    def compile_model(self, imp_model=None):
        print('compile')
        print(imp_model, "imp_model")
        model = imp_model(self.embeddings, self.input_shape, self.target_shape)
        print(model, "model1")
        model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
        self.model = model


    def get_callbacks(self, trains_vals, runs):
        patience = 8
        X_train, y_train, X_val, y_val = trains_vals
        i, run = runs
        self.callbacks = [
            roc_auc_callback(training_data=(X_train, y_train),validation_data=(X_val, y_val)),  # call this before EarlyStopping
            EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),
            CSVLogger('keras-5fold-run-01-v1-epochs.log', separator=',', append=False),
            ModelCheckpoint(
                    'keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check',
                    monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                    save_best_only=True,
                    verbose=1)
        ]


    def get_metrics(self, run_index, model_path, y_val, preds, runs):
        i, run = runs
        LL_run = log_loss(y_val, preds)
        print('\n Fold %d Run %d Log-loss: %.5f' % ((i + 1), (run + 1), LL_run))
        AUC_run = roc_auc_score(y_val, preds)
        print(' Fold %d Run %d AUC: %.5f' % ((i + 1), (run + 1), AUC_run))
        print(' Fold %d Run %d normalized gini: %.5f' % ((i + 1), (run + 1), AUC_run*2-1))
        return [[run_index, LL_run, AUC_run, model_path]]


    def get_fold(self, model=None, folds=4):
        self.skf = StratifiedKFold(n_splits=folds, random_state=1001)
        self.results = []
        self.skf = self.skf.split(self.data, self.label)


    def validate(self, model=None, folds=4, runs=3, KFold_random_state=301):
        skf = StratifiedKFold(n_splits=folds, random_state=KFold_random_state)
        self.results = []
        print("v2")
        for i, (train_index, test_index) in enumerate(skf.split(self.data, self.label)):
            #start_time = timer(None)
            #print(type(self.data), train_index[:12])
            X_train, X_val = self.data[train_index], self.data[test_index]
            y_train, y_val = self.label[train_index], self.label[test_index]
            #train_ids, val_ids = tr_ids[train_index], tr_ids[test_index]
            np.random.seed(123)
            for run in range(runs):
                run_index = f"f_{i}_run_{run}"
                print('\n Fold %d - Run %d\n' % ((i + 1), (run + 1)))
                 self.compile_model(imp_model=model)
                #print("fit",self.model)
                #self.get_callbacks([X_train, y_train, X_val, y_val], [i, run])
                self.model.fit(
                    X_train, y_train, 
                    batch_size=300, 
                    epochs=9, 
                    validation_data=(X_val, y_val),
                    shuffle=True,
                    )
                preds = self.model.predict(X_val, verbose=0)
                self.results += self.get_metrics(run_index, "model_path", y_val, preds, [i, run])
                del self.model
                gc.collect()
        self.prep_results()


    def prep_results(self):
        cols = pd.DataFrame(self.results).T.iloc[0]
        resdf = pd.DataFrame(self.results).T
        resdf.columns = cols
        self.results = resdf.iloc[1:,:]
