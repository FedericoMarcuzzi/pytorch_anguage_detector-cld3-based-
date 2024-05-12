import numpy as np
from nltk import ngrams

class Tokenizer():
    def __init__(self, hash_map_size):
        self.hash_map = HashMap(hash_map_size)

    def get_ngrams_and_freq(self, sentences, max_n):
        hm_size = self.hash_map.get_vocab_size()
        n_sents = len(sentences)
        data = np.zeros((n_sents, max_n, hm_size), dtype=int)
        freq = np.zeros((n_sents, max_n, hm_size))

        for s in range(n_sents):
            for n in range(1, max_n + 1):
                sentence_n_grams = ngrams(sentences[s], n) # computes n-grams
                ids = self.hash_map.hash(list(sentence_n_grams)) # retrieves the ids for each n-gram

                n_grams_ids, n_grams_ids_occ = np.unique(ids, return_counts=True)
                n_grams_ids_freq = n_grams_ids_occ / np.sum(n_grams_ids_occ)

                offset = (n - 1) * hm_size # offset: to match with the correct n-gram embedding in the model.

                # flip to avoid error when reducing bach size
                data[s][n-1][:len(n_grams_ids)] = np.flip(n_grams_ids) + offset 
                freq[s][n-1][:len(n_grams_ids)] = np.flip(n_grams_ids_freq)
            
        # returns two tensors (IDs and n-gram frequencies) of shape (instances, n-gram size, hashmap size).
        return data, freq

class HashMap():
    def __init__(self, map_size):
        self.map_size = map_size
        self.map_w2id = {}

    def get_vocab_size(self):
        return self.map_size

    def _add(self, words):
        for w in words:
            if w not in self.map_w2id:
                self.map_w2id[w] = np.sum([ord(char) for char in w]) % self.map_size # maps each n-gram in a fixed range of IDs

    def hash(self, words):
        self._add(words)
        return [self.map_w2id[w] for w in words]