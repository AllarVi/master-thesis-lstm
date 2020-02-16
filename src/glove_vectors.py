import logging
import os

import numpy as np
from keras.utils import get_file


class GloveVectors:
    @staticmethod
    def get_glove_vectors():
        # Vectors to use
        glove_vectors = '../data/datasets/glove.6B.zip'

        # Download word embeddings if they are not present
        if not os.path.exists(glove_vectors):
            glove_vectors = get_file('glove.6B.zip',
                                     'http://nlp.stanford.edu/data/glove.6B.zip',
                                     cache_dir="../data/")

        glove_vectors_100d = '../data/datasets/glove.6B.100d.txt'

        if not os.path.exists(glove_vectors_100d) and os.path.exists(glove_vectors):
            os.system(f'unzip {glove_vectors} -d ../data/datasets/')

        # Load in unzipped file
        glove = np.loadtxt(glove_vectors_100d, dtype='str', comments=None)

        logging.info(f"glove shape: {glove.shape}")

        # Now we separated into the words and the vectors.
        vectors = glove[:, 1:].astype('float')
        words = glove[:, 0]

        del glove

        logging.info(f"Example of word vector at index 100: {vectors[100]}")
        logging.info(f'Example of word at index 100: "{words[100]}"')

        logging.info(f"Word vectors count: {vectors.shape}")

        return vectors, words
