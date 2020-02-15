import logging

from tensorflow.python.client import device_lib

from src.data_cleaner import DataCleaner
from src.logging_config import LoggingConfig
from src.patent_data_loader import PatentDataLoader
from src.plots import Plots
from src.text_to_sequences import TextToSequences


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
    num_words, \
    word_counts, \
    abstracts, \
    sequences = text_to_sequences.make_sequences(TRAINING_LENGTH,
                                                 lower=True,
                                                 filters=filters)

    n = 3

    text_to_sequences.find_answer(n)

    logging.info(f"Most common words: {sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]}")


if __name__ == '__main__':
    main()
