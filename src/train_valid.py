import numpy as np
from sklearn.utils import shuffle

RANDOM_STATE = 50
TRAIN_FRACTION = 0.7


class TrainValid:

    def __init__(self, features, labels) -> None:
        self.features = features
        self.labels = labels

    def create_train_valid(self,
                           num_words,
                           train_fraction=TRAIN_FRACTION):
        """Create training and validation features and labels."""

        # Randomly shuffle features and labels
        features, labels = shuffle(self.features, self.labels, random_state=RANDOM_STATE)

        # Decide on number of samples for training
        train_end = int(train_fraction * len(labels))

        train_features = np.array(features[:train_end])
        valid_features = np.array(features[train_end:])

        train_labels = labels[:train_end]
        valid_labels = labels[train_end:]

        # Convert to arrays
        X_train, X_valid = np.array(train_features), np.array(valid_features)

        # Using int8 for memory savings
        y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
        y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

        # One hot encoding of labels
        for example_index, word_index in enumerate(train_labels):
            y_train[example_index, word_index] = 1

        for example_index, word_index in enumerate(valid_labels):
            y_valid[example_index, word_index] = 1

        # Memory management
        import gc
        gc.enable()
        del features, labels, train_features, valid_features, train_labels, valid_labels
        gc.collect()

        return X_train, X_valid, y_train, y_valid
