from keras.callbacks import EarlyStopping, ModelCheckpoint

SAVE_MODEL = True
MODEL_DIR = '../models/'


class Callbacks:

    @staticmethod
    def make_callbacks(model_name, save=SAVE_MODEL):
        """Make list of callbacks for training"""
        callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

        if save:
            callbacks.append(
                ModelCheckpoint(
                    f'{MODEL_DIR}{model_name}.h5',
                    save_best_only=True,
                    save_weights_only=False))

        return callbacks
