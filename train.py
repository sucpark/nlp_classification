import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from utils import *
from model.metric import evaluate, acc, LSR
from model.data import DAdataset
from model.net import LSTMClassifier
from model.utils import sent_tokenize, stemming, preprocess_text
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Training nlp classification model")
parser.add_argument('--data_dir', default='dataset', help="Directory containing data")
parser.add_argument('--save_dir', default='experiment')
parser.add_argument('--data_name', default='SDAC')

parser_for_training = parser.add_argument_group(title='Training')
parser_for_training.add_argument('--epoch', default=100, type=int)
parser_for_training.add_argument('--batch_size', dest='n_batch', default=64, type=int)
parser_for_training.add_argument('--max_len', default=256, type=int)
parser_for_training.add_argument('--learning_rate', dest='lr', default=1e-5, type=float)
parser_for_training.add_argument('--embedding_dim', default=128, type=int)
parser_for_training.add_argument('--hidden_size', default=256, type=int)
parser_for_training.add_argument('--layer_size', dest='n_layers', default=2, type=int)
parser_for_training.add_argument('--dropout', default=0.5, type=float)
parser_for_training.add_argument('--summary_step', default=10000, type=int)
parser_for_training.add_argument('--save_pig', default=False, type=bool)

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir) / args.data_name
    save_dir = Path(args.save_dir) / args.data_name
    train_data_name = 'sw_train.txt'
    valid_data_name = 'sw_val.txt'
    token2idx_name = 'word2idx.json'
    label2idx_name = 'label2idx.json'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(data_dir / token2idx_name, 'r') as f:
        token2idx = json.load(f)

    with open(data_dir / label2idx_name, 'r') as f:
        label2idx = json.load(f)

    idx2token = {i: t for t, i in token2idx.items()}
    idx2label = {i: l for l, i in label2idx.items()}

    train_data = pd.read_csv(data_dir / train_data_name, header=None, sep='|', names=['speaker', 'utterance', 'tag'])
    valid_data = pd.read_csv(data_dir / valid_data_name, header=None, sep='|', names=['speaker', 'utterance', 'tag'])

    x_train, y_train = train_data['utterance'], train_data['tag']
    x_valid, y_valid = valid_data['utterance'], valid_data['tag']

    text_preprocess_pipeline = [sent_tokenize, stemming]
    x_train = x_train.apply(preprocess_text, processing_function_list=text_preprocess_pipeline)
    x_valid = x_valid.apply(preprocess_text, processing_function_list=text_preprocess_pipeline)

    x_train = list(convert_token_to_idx(x_train, token2idx))
    x_valid = list(convert_token_to_idx(x_valid, token2idx))
    y_train = list(convert_label_to_idx(y_train, label2idx))
    y_valid = list(convert_label_to_idx(y_valid, label2idx))

    tr_ds = DAdataset(x_train, y_train)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.n_batch, collate_fn=tr_ds.collate_fn, drop_last=True)
    val_ds = DAdataset(x_valid, y_valid)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.n_batch, collate_fn=val_ds.collate_fn, drop_last=True)

    vocab_size = len(token2idx)
    output_size = len(label2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model = LSTMClassifier(output_size, args.hidden_size, vocab_size, args.n_layers,
                           args.embedding_dim, device, dropout=args.dropout, bidirectional=True)
    model.to(device)

    loss_fn = LSR(epsilon=0.1, num_classes=output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    writer = SummaryWriter(f'{save_dir}/runs')
    checkpoint_manager = CheckpointManager(save_dir)
    summary_manager = SummaryManager(save_dir)

    best_val_loss = 1e+10
    train_acc, valid_acc = [], []
    train_losses, valid_losses = [], []

    for epoch in tqdm(range(args.epoch), desc='epochs'):
        tr_loss = 0
        tr_acc = 0
        model.train()

        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)
            optimizer.zero_grad()
            y_hat_mb = model(x_mb)
            loss = loss_fn(y_hat_mb, y_mb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mb_acc = acc(y_hat_mb, y_mb)

            # clip_gradient(model, 1e-1)

            tr_loss += loss.item()
            tr_acc += mb_acc.item()

            if (epoch * len(tr_dl) + step) % args.summary_step == 0:
                val_loss = evaluate(model, val_dl, {'loss': loss_fn}, device)['loss']
                writer.add_scalars('loss', {'train': tr_loss / (step + 1), 'val': val_loss}, epoch * len(tr_dl) + step)
                tqdm.write('global_Step: {:3}, tr_loss: {:3f}, val_loss: {:3f}'.format(epoch * len(tr_dl) + step,
                                                                                       tr_loss / (step + 1),
                                                                                       val_loss))
                model.train()

        else:
            tr_loss /= (step + 1)
            tr_acc /= (step + 1)
            train_losses.append(tr_loss)
            train_acc.append(tr_acc)

            tr_sum = {'loss': tr_loss, 'acc': tr_acc}
            val_sum = evaluate(model, val_dl, {'loss': loss_fn, 'acc': acc}, device)
            tqdm.write('epoch : {}, tr_loss: {: 3f}, val_loss: '
                       '{:.3f}, tr_acc: {:.2%}, val_acc: {:.2%}'.format(epoch + 1, tr_sum['loss'], val_sum['loss'],
                                                                        tr_sum['acc'], val_sum['acc']))

            val_loss = val_sum['loss']
            valid_losses.append(val_loss)
            valid_acc.append(val_sum['acc'])
            is_best = val_loss < best_val_loss

            if is_best:
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict()
                }
                summary = {'tr': tr_sum, 'val': val_sum}
                summary_manager.update(summary)
                summary_manager.save('summary.json')
                checkpoint_manager.save_checkpoint(state, 'best.tar')
                best_val_loss = val_loss
    if args.save_pig:
        best_point = valid_losses.index(min(valid_losses)) + 1
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
        plt.axvline(best_point, linestyle='dotted', color='r', label='Best Model Checkpoint')
        plt.title('Loss Curve')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig(f'{save_dir}/loss_plot.png', bbox_inches='tight')

        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
        plt.plot(range(1, len(valid_acc) + 1), valid_acc, label='Validation Accuracy')
        plt.axvline(best_point, linestyle='--', color='r', label='Best Model Checkpoint')
        plt.title('Accuracy Plot')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig(f'{save_dir}/accuracy_plot.png', bbox_inches='tight')

