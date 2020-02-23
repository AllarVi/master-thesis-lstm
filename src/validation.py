from keras.models import load_model

MODEL_DIR = '../models/'


class Validation:

    @staticmethod
    def load_and_evaluate(X_valid, y_valid, model_name, return_model=False):
        """Load in a trained model and evaluate with log loss and accuracy"""

        model = load_model(f'{MODEL_DIR}{model_name}.h5')
        r = model.evaluate(X_valid, y_valid, batch_size=2048, verbose=1)

        valid_crossentropy = r[0]
        valid_accuracy = r[1]

        print(f'Cross Entropy: {round(valid_crossentropy, 4)}')
        print(f'Accuracy: {round(100 * valid_accuracy, 2)}%')

        if return_model:
            return model
