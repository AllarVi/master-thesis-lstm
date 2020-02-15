import logging

from tensorflow.python.client import device_lib

from src.data_cleaner import DataCleaner
from src.logging_config import LoggingConfig
from src.patent_data_loader import PatentDataLoader
from src.plots import Plots


def main():
    LoggingConfig.setup()

    # Get available devices (CPUs and GPUs)
    logging.debug(device_lib.list_local_devices())

    [data, original_abstracts] = PatentDataLoader.get_original_abstracts()

    Plots.neural_network_patents_over_time(data)
    Plots.neural_network_patents_by_year(data)

    # Data cleaning
    formatted = DataCleaner.get_formatted(original_abstracts)


if __name__ == '__main__':
    main()
