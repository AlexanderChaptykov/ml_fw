from keras.models import Model
from keras.layers import *
from layers import Attention
from keras.optimizers import Adam


def base_cnn(embeddings_matrix, input_len, target_len):
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
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-2), metrics=['accuracy'])
    return model


def cnn2(embeddings_matrix, input_len, target_len):
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
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-2), metrics=['accuracy'])
    return model


def attention(embeddings_matrix, input_len, target_len):
    inp = Input(shape=(input_len,))
    x = Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1], weights=[embeddings_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Attention(input_len)(x)
    x = Dense(target_len, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-2), metrics=['accuracy'])
    return model
