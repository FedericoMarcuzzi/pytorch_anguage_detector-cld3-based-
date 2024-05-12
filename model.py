import torch
import torch.nn as nn
import torch.optim as optim
from data_manager import Batcher

class LanguageDetector(nn.Module):
    def __init__(self, max_n_grams, vocab_size, embedding_size, hidden_size, output_size):
        super(LanguageDetector, self).__init__()
        self.max_n_grams = max_n_grams
        self.embedding_size = embedding_size

        self.embeddings = nn.Embedding(vocab_size * self.max_n_grams, self.embedding_size)
        self.hidden_layer = nn.Linear(self.embedding_size * self.max_n_grams, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, input_idxs, input_freq):
        embeddings = self.embeddings(input_idxs)
        weighted_mean_embeddings = torch.sum(embeddings * input_freq, dim=2) # embeddings' weighted mean based on the n-grams frequency.
        concatenated_embeddings = weighted_mean_embeddings.reshape((input_freq.shape[0], self.embedding_size * self.max_n_grams)) # like a concatenation of embedding, each for a specific n-gram size, e.g., 1-gram + 2-gram + 3-gram, ...
        hidden_output = self.relu(self.hidden_layer(concatenated_embeddings))
        output = self.softmax(self.output_layer(hidden_output))

        return output

class LanguageDetectorManager():
    def __init__(self, max_n_grams, hash_map_size, embedding_size, hidden_size, output_size, learning_rate):
        self.model = LanguageDetector(max_n_grams, hash_map_size, embedding_size, hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.eval_sets = {}

    def fit(self, train_data, train_freq, train_labels, batch_size):
        self.batch_size = batch_size
        self.train_batches = Batcher(train_data, train_freq, train_labels, self.batch_size)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_acc = 0
            for i in range(len(self.train_batches)):
                batch_idxs, batch_freq, batch_labels = self.train_batches[i]

                self.optimizer.zero_grad()
                output = self.model(batch_idxs, batch_freq)
                loss = self.criterion(output, batch_labels)
                loss.backward()
                self.optimizer.step()

                train_acc += torch.sum(torch.argmax(output, dim=1) == batch_labels)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch num: [{epoch + 1} / {num_epochs}], Loss: {loss.item()}, Train acc: {train_acc / self.train_batches.n_insts}')

    def set_eval_data(self, eval_name, eval_data, eval_freq, eval_labels):
        self.eval_sets[eval_name] = Batcher(eval_data, eval_freq, eval_labels, self.batch_size)

    def eval(self, eval_name):
        if eval_name not in self.eval_sets:
            print("call object.<set_eval_data> first.")
            return
        
        eval_batches = self.eval_sets[eval_name]

        test_acc = 0
        for i in range(len(eval_batches)):
            batch_idxs, batch_freq, batch_labels = eval_batches[i]
            output = self.model(batch_idxs, batch_freq)
            test_acc += torch.sum(torch.argmax(output, dim=1) == batch_labels)

        print(f'{eval_name.capitalize()} Acc: {test_acc / eval_batches.n_insts}')