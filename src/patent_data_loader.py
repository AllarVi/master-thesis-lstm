import logging

import pandas as pd


class PatentDataLoader:

    @staticmethod
    def get_original_abstracts():
        # Read in data
        data = pd.read_csv('../data/neural_network_patent_query_updated.csv', parse_dates=['patent_date'])

        # Extract abstracts
        original_abstracts = list(data['patent_abstract'])

        logging.info(f"Loaded {len(original_abstracts)} original abstracts")
        # logging.info(data.head())

        return [data, original_abstracts]
