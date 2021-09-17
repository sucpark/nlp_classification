import argparse
import pandas as pd
from utils import *
from model.metric import predicate
from model.data import DAdataset
from model.net import LSTMClassifier
from model.utils import sent_tokenize, stemming, preprocess_text
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Evaluating nlp classification model")
parser.add_argument('--data_dir', default='dataset', help="Directory containing data")
parser.add_argument('--save_dir', default='experiment', help="Directory containing experiment results")
parser.add_argument('--data_name', default='SDAC')

parser_for_training = parser.add_argument_group(title='Evaluating')
parser_for_training.add_argument('--learning_rate', dest='lr', default=1e-5, type=float)
parser_for_training.add_argument('--embedding_dim', default=512, type=int)
parser_for_training.add_argument('--hidden_size', default=512, type=int)
parser_for_training.add_argument('--layer_size', dest='n_layers', default=2, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir) / args.data_name
    save_dir = Path(args.save_dir) / args.data_name
    test_data_name = 'sw_test.csv'
    token2idx_name = 'word2idx.json'
    label2idx_name = 'label2idx.json'

    with open(data_dir / token2idx_name, 'r') as f:
        token2idx = json.load(f)

    with open(data_dir / label2idx_name, 'r') as f:
        label2idx = json.load(f)

    idx2label = {i: l for l, i in label2idx.items()}

    batch_size = 1
    vocab_size = len(token2idx)
    output_size = len(label2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print('GPU is available')
    model = LSTMClassifier(output_size, args.hidden_size, vocab_size, args.n_layers,
                           args.embedding_dim, device, bidirectional=True)
    model.to(device)

    writer = SummaryWriter(f'{save_dir}/runs')
    checkpoint_manager = CheckpointManager(save_dir)
    summary_manager = SummaryManager(save_dir)
    
    print('Load pretrained model')
    ckpt = checkpoint_manager.load_checkpoint('best.tar')
    model.load_state_dict(ckpt['model_state_dict'])

    test_data = pd.read_csv(data_dir / test_data_name, header=None, sep=',', names=['speaker', 'utterance'])
    x_test = test_data['utterance']
    y_fake = [0] * len(x_test)
    text_preprocess_pipeline = [sent_tokenize, stemming]
    x_test = x_test.apply(preprocess_text, processing_function_list=text_preprocess_pipeline)
    x_test = list(convert_token_to_idx(x_test, token2idx))
    test_ds = DAdataset(x_test, y_fake)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, collate_fn=test_ds.collate_fn)
    predicates = predicate(model, test_dl, device)

    y_pred = [idx2label[p] for p in predicates]
    test_data['prediction'] = y_pred
    test_data.to_csv(save_dir / test_data_name, sep=',', index=False, header=False)
