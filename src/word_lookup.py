import logging

import numpy as np


class WordLookup:

    def __init__(self, unique_words_count, word_idx, idx_word) -> None:
        self.unique_words_count = unique_words_count
        self.word_idx = word_idx
        self.idx_word = idx_word
        self.embedding_matrix = []

    def get_word_lookup(self, words, vectors):
        word_lookup = {word: vector for word, vector in zip(words, vectors)}

        embedding_matrix = np.zeros((self.unique_words_count, vectors.shape[1]))

        not_found = 0

        for i, word in enumerate(self.word_idx.keys()):
            # Look up the word embedding
            vector = word_lookup.get(word, None)

            # Record in matrix
            if vector is not None:
                embedding_matrix[i + 1, :] = vector
            else:
                not_found += 1

        logging.info(f'There were {not_found} words without pre-trained embeddings.')

        import gc
        gc.enable()
        del vectors
        gc.collect()

        # Each word is represented by 100 numbers with a number of words that can't be found.
        # We can find the closest words to a given word in embedding space using the cosine distance.
        # This requires first normalizing the vectors to have a magnitude of 1.
        # Normalize and convert nan to 0
        embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))
        self.embedding_matrix = np.nan_to_num(embedding_matrix)

    def find_closest(self, query, n=10):
        """Find closest words to a query word in embeddings"""

        idx = self.word_idx.get(query, None)
        # Handle case where query is not in vocab
        if idx is None:
            logging.info(f'{query} not found in vocab.')
            return
        else:
            vec = self.embedding_matrix[idx]
            # Handle case where word doesn't have an embedding
            if np.all(vec == 0):
                logging.info(f'{query} has no pre-trained embedding.')
                return
            else:
                # Calculate distance between vector and all others
                dists = np.dot(self.embedding_matrix, vec)

                # Sort indexes in reverse order
                idxs = np.argsort(dists)[::-1][:n]
                sorted_dists = dists[idxs]
                closest = [self.idx_word[i] for i in idxs]

        logging.info(f'Query: {query}\n')
        max_len = max([len(i) for i in closest])
        # Print out the word and cosine distances
        for word, dist in zip(closest, sorted_dists):
            logging.info(f'Word: {word:15} Cosine Similarity: {round(dist, 4)}')
