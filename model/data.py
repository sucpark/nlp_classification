import torch

class DAdataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        assert len(self.inputs) == len(self.labels)
        return len(self.labels)
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.inputs[index]),
            torch.tensor(self.labels[index]),
        )
    def collate_fn(self, batch):
        inputs, labels = list(zip(*batch))
        
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        
        batch = [
            inputs,
            labels
        ]
        return batch