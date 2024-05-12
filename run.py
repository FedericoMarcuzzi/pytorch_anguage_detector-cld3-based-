from misc import set_seed
from tokenizer import Tokenizer
from model import LanguageDetectorManager

import numpy as np
from sklearn.model_selection import train_test_split


SEED = 7
TRAIN_SIZE = 0.5

HASH_MAP_SIZE = 256
MAX_N_GRAMS = 5

LEARNING_RATE = 1e-2
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 32
BATCH_SIZE = 100
NUM_EPOCHS = 20

set_seed(SEED)

sentences = ["set di dati di addestramento", "set di dati di test", "conjunto de datos de entrenamiento", "conjunto de datos de prueba", "training dataset", "test dataset", "Trainingsdatensatz", "Testdatensatz", "ensemble de données de formation", "ensemble de données de test"]
labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]) # 0 : it, 1 : spa, 2 : en, 3: de, 4 : fr.

tokenizer = Tokenizer(HASH_MAP_SIZE)
data, freq = tokenizer.get_ngrams_and_freq(sentences, MAX_N_GRAMS)

train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=TRAIN_SIZE, random_state=SEED, stratify=labels)
train_data, train_freq, train_labels = data[train_idx], freq[train_idx], labels[train_idx]
test_data, test_freq, test_labels = data[test_idx], freq[test_idx], labels[test_idx]

OUTPUT_SIZE = np.unique(labels).shape[0]

ldm = LanguageDetectorManager(MAX_N_GRAMS, HASH_MAP_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE)
ldm.fit(train_data, train_freq, train_labels, BATCH_SIZE)
ldm.set_eval_data("test set", test_data, test_freq, test_labels)
ldm.train(NUM_EPOCHS)
ldm.eval("test set")