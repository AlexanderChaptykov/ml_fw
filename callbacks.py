from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score


class RocAucEvaluation(Callback):
    """
    Считает рок_аук из сета который получается при автоматическом разделении модели,
    либо из заданного сета
    validation_data - tuple or list в котором 2 элемента, первый это X а второй y

    """

    def __init__(self, validation_data=[], interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        if validation_data:
            self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.validation_data[0], verbose=0)
            score = roc_auc_score(self.validation_data[1], y_pred, average='micro')
            print(f"ROC-AUC micro avg - epoch: {epoch + 1} - score: {score}\n")




def get_callbacks(X_train, X_test, y_train, y_test, patience=2,
                  model_path='temp_.hdf5'):
    checkpoint = ModelCheckpoint(model_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    callbacks_list = [
        EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=1),
        RocAucEvaluation(),
        CSVLogger('keras.log', separator=',', append=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    ]
    return callbacks_list
