from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1, type_="Train"):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.type_ = type_

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred, average='micro')
            print(f"ROC-AUC micro avg {self.type_} - epoch: {epoch+1} - score: {score}\n")


def get_callbacks(train_X,
                  train_y,
                  val_X,
                  val_y,
                  model_path=f'temp_{i}.hdf5'
                  ):
    checkpoint = ModelCheckpoint(model_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=1)
    ra_val = RocAucEvaluation(validation_data=(val_X, val_y), interval=1, type_="val")
    ra_train = RocAucEvaluation(validation_data=(train_X, train_y), interval=1, type_="train")
    logger = CSVLogger('keras.log', separator=',', append=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    callbacks_list = [ra_train, ra_val, early, reduce_lr, logger]
    return callbacks_list
