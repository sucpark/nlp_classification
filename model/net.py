import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, n_layers, embedding_dim, device,
                 dropout=0.0, bidirectional=False):
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

        self.init_word_embedding(padding_idx=0)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def init_word_embedding(self, padding_idx):

        torch.nn.init.kaiming_normal_(self.embeddings.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.embeddings.weight.data[padding_idx].uniform_(0, 0)

    def init_hidden(self, batch_size):
        h_0 = Variable(torch.randn(self.n_direction * self.n_layers, batch_size, self.hidden_size)).to(self.device)
        c_0 = Variable(torch.randn(self.n_direction * self.n_layers, batch_size, self.hidden_size)).to(self.device)

        return h_0, c_0

    def forward(self, input_sentence):
        inputs = self.embeddings(input_sentence)
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)
        # inputs_len = torch.tensor(list(map(sum, input_sentence != 0)))
        # inputs_packed = pack_padded_sequence(inputs, inputs_len, batch_first=True, enforce_sorted=False)
        # output_packed, (final_hidden, final_cell) = self.lstm(inputs_packed, hidden)
        # output_padded, outputs_len = pad_packed_sequence(output_packed, batch_first=True)
        # output_forward = output_padded[range(len(output_padded)), outputs_len-1, :self.hidden_size*self.n_direction]
        # output_backward = output_padded[:, 0, self.hidden_size*self.n_direction:]
        # output = torch.cat((output_forward, output_backward), dim=1)

        _, (final_hidden, final_cell) = self.lstm(inputs, hidden)
        output = torch.cat((final_hidden[-2, :, :], final_hidden[-1, :, :]), dim=1)
        output = self.relu(output)
        output = self.dropout(output)
        fc1 = self.fc1(output)
        fc2 = self.fc2(fc1)

        return fc2

