import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, n_layers, embedding_dim, device,
                 dropout=0.5, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        self.n_direction = 2 if bidirectional else 1

        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers,
                            dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(self.n_direction * self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.zeros(self.n_direction * self.n_layers, batch_size, self.hidden_size)).to(self.device)
        c_0 = Variable(torch.zeros(self.n_direction * self.n_layers, batch_size, self.hidden_size)).to(self.device)

        return h_0, c_0

    def forward(self, input_sentence):
        inputs = self.embeddings(input_sentence)
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)
        _, (final_hidden, final_cell) = self.lstm(inputs, hidden)

        output = torch.cat((final_hidden[-2, :, :], final_hidden[-1, :, :]), dim=1)
        output = self.relu(output)
        fc1 = self.fc1(output)
        fc1 = self.dropout(fc1)
        fc2 = self.fc2(fc1)
        return fc2

