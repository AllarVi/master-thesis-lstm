import logging
import sys

from keras.utils import plot_model
from tensorflow.python.client import device_lib

from src.build_model import make_word_level_model
from src.callbacks import Callbacks
from src.data_cleaner import DataCleaner
from src.glove_vectors import GloveVectors
from src.logging_config import LoggingConfig
from src.patent_data_loader import PatentDataLoader
from src.plots import Plots
from src.text_to_sequences import TextToSequences
from src.train_valid import TrainValid
from src.validation import Validation
from src.word_lookup import WordLookup

BATCH_SIZE = 2048
EPOCHS = 150
VERBOSE = 1
LSTM_CELLS = 64
MODEL_NAME = 'pre-trained-rnn'
MODEL_DIR = '../models/'


def check_sizes(gb_min=1):
    for x in globals():
        size = sys.getsizeof(eval(x)) / 1e9
        if size > gb_min:
            print(f'Object: {x:10}\tSize: {size} GB.')


def main():
    LoggingConfig.setup()

    # Get available devices (CPUs and GPUs)
    logging.debug(device_lib.list_local_devices())

    [data, original_abstracts] = PatentDataLoader.get_original_abstracts()

    Plots.neural_network_patents_over_time(data)
    Plots.neural_network_patents_by_year(data)

    # Data cleaning
    formatted = DataCleaner.get_formatted(original_abstracts)

    # Convert Text to Sequences
    TRAINING_LENGTH = 50
    filters = '!"#$%&()*+/:<=>@[\\]^_`{|}~\t\n'
    text_to_sequences = TextToSequences(formatted)

    word_idx, \
    unique_words_count, \
    word_counts, \
    abstracts, \
    sequences, \
    features, \
    labels, \
    idx_word = text_to_sequences.make_sequences(TRAINING_LENGTH,
                                                lower=True,
                                                filters=filters)

    n = 3

    text_to_sequences.find_answer(n)

    logging.info(f"Most common words: {sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]}")

    # Encoding of Labels
    train_valid = TrainValid(features, labels)

    X_train, X_valid, y_train, y_valid = train_valid.create_train_valid(unique_words_count)

    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")

    # 1e9 = 1000000000 (a billion)
    # 1 MB = 1e6
    # 1 GB = 1e9
    # sys.getsizeof() returns size in bytes
    # 1 byte = 8 bits, bits are either 0's or 1's
    logging.info(f"Size of y_train in GB: {sys.getsizeof(y_train) / 1e9}")

    check_sizes(gb_min=1)

    vectors, words = GloveVectors.get_glove_vectors()

    word_lookup = WordLookup(unique_words_count, word_idx, idx_word)
    embedding_matrix = word_lookup.get_word_lookup(words, vectors)

    # word_lookup.find_closest("the")
    # word_lookup.find_closest("neural")
    # word_lookup.find_closest(".")
    # word_lookup.find_closest("wonder")
    # word_lookup.find_closest("dnn")

    model = make_word_level_model(
        unique_words_count,
        embedding_matrix=embedding_matrix,
        lstm_cells=LSTM_CELLS,
        trainable=False,
        lstm_layers=1)

    logging.info("Plotting model architecture to console")
    model.summary()

    logging.info(f"Saving model architecture to file {MODEL_DIR}{MODEL_NAME}.png")
    plot_model(model, to_file=f'{MODEL_DIR}{MODEL_NAME}.png', show_shapes=True)

    callbacks = Callbacks.make_callbacks(MODEL_NAME)

    # Depending on your machine, this may take several hours to run.

    logging.info("Starting training...")

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid))

    logging.info("Model loading and validation")
    model = Validation.load_and_evaluate(MODEL_NAME, return_model=True)


if __name__ == '__main__':
    main()
