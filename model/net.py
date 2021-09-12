import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_dim, bidirectional=False, weights=None):
        super(LSTMClassifier, self).__init__()
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.embeddings.weight = nn.Parameter(weights, requires_grad=True, )
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sentence, batch_size=None):
        inputs = self.embeddings(input_sentence)
        inputs = inputs.permute(1, 0, 2)
        
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda()
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda()
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
            
        outputs, (final_hidden_state, final_cell_state) = self.lstm(inputs, (h_0, c_0))
        final_outputs = self.fc(final_hidden_state[-1])
        
        return final_outputs