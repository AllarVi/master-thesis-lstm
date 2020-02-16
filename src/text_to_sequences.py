import logging

from keras.preprocessing.text import Tokenizer


class TextToSequences:

    def __init__(self, texts):
        self.texts = texts
        self.idx_word = []
        self.training_seq = []
        self.labels = []

    def find_answer(self, index):
        """Find label corresponding to features for index in training data"""

        # Find features and label
        feats = ' '.join(self.idx_word[i] for i in self.training_seq[index])
        answer = self.idx_word[self.labels[index]]

        logging.info(f'Features: {feats}')
        logging.info(f'Label: {answer}')

    def make_sequences(self,
                       training_length=50,
                       lower=True,
                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        """Turn a set of texts into sequences of integers"""

        # Create the tokenizer object and train on texts
        tokenizer = Tokenizer(lower=lower, filters=filters)
        tokenizer.fit_on_texts(self.texts)

        # Create look-up dictionaries and reverse look-ups
        word_idx = tokenizer.word_index
        self.idx_word = tokenizer.index_word
        unique_words_count = len(word_idx) + 1
        word_counts = tokenizer.word_counts

        logging.info(f'There are {unique_words_count} unique words.')

        # Convert text to sequences of integers
        sequences = tokenizer.texts_to_sequences(self.texts)

        # Limit to sequences with more than training length tokens
        seq_lengths = [len(x) for x in sequences]
        over_idx = [
            i for i, l in enumerate(seq_lengths) if l > (training_length + 20)
        ]

        new_texts = []
        new_sequences = []

        # Only keep sequences with more than training length tokens
        for i in over_idx:
            new_texts.append(self.texts[i])
            new_sequences.append(sequences[i])

        # Iterate through the sequences of tokens
        for seq in new_sequences:

            # Create multiple training examples from each sequence
            for i in range(training_length, len(seq)):
                # Extract the features and label
                extract = seq[i - training_length:i + 1]

                # Set the features and label
                self.training_seq.append(extract[:-1])
                self.labels.append(extract[-1])

        logging.info(f'There are {len(self.training_seq)} training sequences.')

        # Return everything needed for setting up the model
        return word_idx, \
               unique_words_count, \
               word_counts, \
               new_texts, \
               new_sequences, \
               self.training_seq, \
               self.labels
