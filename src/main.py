import logging
import sys

from tensorflow.python.client import device_lib

from src.data_cleaner import DataCleaner
from src.glove_vectors import GloveVectors
from src.logging_config import LoggingConfig
from src.patent_data_loader import PatentDataLoader
from src.plots import Plots
from src.text_to_sequences import TextToSequences
from src.train_valid import TrainValid
from src.word_lookup import WordLookup


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

    logging.info(f"{sys.getsizeof(y_train) / 1e9}")

    check_sizes(gb_min=1)

    vectors, words = GloveVectors.get_glove_vectors()

    word_lookup = WordLookup(unique_words_count, word_idx, idx_word)
    word_lookup.get_word_lookup(words, vectors)

    # word_lookup.find_closest("the")
    # word_lookup.find_closest("neural")
    # word_lookup.find_closest(".")
    # word_lookup.find_closest("wonder")
    # word_lookup.find_closest("dnn")


if __name__ == '__main__':
    main()
