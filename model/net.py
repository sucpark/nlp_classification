import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, n_layers, embedding_dim, device,
                 dropout=0.7, bidirectional=False, weights=None):
        super(LSTMClassifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.dropout = nn.Dropout(dropout)
        if weights is not None:
            assert weights.shape == (vocab_size, embedding_dim), "The shape of embedding matrix is not matched"
            self.embeddings.load_state_dict({'weight': weights})

        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_sentence):
        inputs = self.embeddings(input_sentence)

        if self.bidirectional:
            h_0 = Variable(torch.ones(2 * self.n_layers, self.batch_size, self.hidden_size), requires_grad=True).to(
                self.device)
            c_0 = Variable(torch.ones(2 * self.n_layers, self.batch_size, self.hidden_size), requires_grad=True).to(
                self.device)
        else:
            h_0 = Variable(torch.oness(self.n_layers, self.batch_size, self.hidden_size), requires_grad=True).to(
                self.device)
            c_0 = Variable(torch.oness(self.n_layers, self.batch_size, self.hidden_size), requires_grad=True).to(
                self.device)

        outputs, (final_hidden_state, final_cell_state) = self.lstm(inputs, (h_0, c_0))
        final_outputs = self.fc(self.dropout(final_hidden_state[-1]))

        return final_outputs
