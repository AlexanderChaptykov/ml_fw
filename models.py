from keras.models import Model
from keras.layers import *
from .layers import Attention
from keras.optimizers import Adam
from keras.regularizers import l1

def base_cnn(embeddings_matrix, input_len, target_len, metrics=['accuracy']):
    """{'embedding_matrix':embedding_matrix, 'MAX_SEQUENCE_LENGTH':MAX_SEQUENCE_LENGTH}"""
    embedding_layer = Embedding(embeddings_matrix.shape[0],
                                embeddings_matrix.shape[1],
                                weights=[embeddings_matrix],
                                input_length=input_len,
                                trainable=False)
    sequence_input = Input(shape=(input_len,), dtype='int32')
    #print(embedding_layer.get_config)
    #print(sequence_input)
    embedded_sequences = embedding_layer(sequence_input)
    #print(embedded_sequences)
    x = Conv1D(embeddings_matrix.shape[1], 2, activation='relu')(embedded_sequences)
    x = MaxPooling1D(3)(x)
    x = Conv1D(embeddings_matrix.shape[1], 2, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(embeddings_matrix.shape[1], activation='relu')(x)
    preds = Dense(target_len, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metrics)
    return model


def cnn2(embeddings_matrix, input_len, target_len, metrics=['accuracy']):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36
    inp = Input(shape=(input_len,))
    x = Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1], weights=[embeddings_matrix])(inp)
    x = Reshape((embeddings_matrix.shape[0], embeddings_matrix.shape[1], 1))(x)
    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embeddings_matrix.shape[1]),
                                    kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(embeddings_matrix.shape[0] - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
    outp = Dense(target_len, activation="sigmoid")(z)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-2), metrics=metrics)
    return model



def cnn3(embedding_matrix, input_len, target_len, metrics=['accuracy']):
    filter_sizes = [1,2,3,5]
    num_filters = 36
    embed_size = embedding_matrix.shape[1]

    inp = Input(shape=(input_len,))
    x = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Reshape((input_len, embed_size, 1))(x)
    #add
    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                    kernel_initializer='he_normal', activation='elu')(x)
        conv = MaxPool2D(pool_size=(input_len - filter_sizes[i] + 1, 1))(conv)
        #maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))
        #conv = BatchNormalization()(conv)
        #conv = Dropout(0.6)(conv)
        maxpool_pool.append(conv)
    z = Concatenate(axis=1)(maxpool_pool)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    z = Flatten()(z)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    outp = Dense(target_len, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=metrics)
    return model


def attention(embedding_matrix, input_len, target_len, metrics=['accuracy']):
    embed_size = embedding_matrix.shape[1]
    inp = Input(shape=(input_len,))
    x = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Attention(input_len)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(target_len, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metrics)
    return model


def elmo_model(embed_size, input_len, target_len, metrics=['accuracy']):
    inp = Input(shape=(input_len, embed_size))
    x = SpatialDropout1D(0.10)(inp)
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.10, recurrent_dropout=0.10))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    out = Dense(target_len, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=metrics)
    return model
