import logging
import re


class DataCleaner:

    @staticmethod
    def format_patent(patent):
        """Add spaces around punctuation and remove references to images/citations."""

        # Add spaces around punctuation
        patent = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', patent)

        # Remove references to figures
        patent = re.sub(r'\((\d+)\)', r'', patent)

        # Remove double spaces
        patent = re.sub(r'\s\s', ' ', patent)
        return patent

    @staticmethod
    def remove_spaces(patent):
        """Remove spaces around punctuation"""
        patent = re.sub(r'\s+([.,;?])', r'\1', patent)

        return patent

    @staticmethod
    def get_formatted(original_abstracts):
        formatted = []

        # Iterate through all the original abstracts
        for a in original_abstracts:
            formatted.append(DataCleaner.format_patent(a))

        logging.info(f"Number of formatted abstracts: {len(formatted)}")

        return formatted
