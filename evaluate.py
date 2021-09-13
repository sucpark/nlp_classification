import os
import sys
import argparse
import pandas as pd
from utils import *
from model.metric import evaluate, predicate, acc, LSR
from model.data import DAdataset
from model.net import LSTMClassifier
from model.utils import sent_tokenize, stemming, preprocess_text
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, '..')

if __name__ == "__main__":

    home_dir = Path('..')
    data_dir = home_dir / 'dataset' / 'SDAC'
    save_dir = home_dir / 'experiment' / 'SDAC'
    train_data_name = 'sw_train.txt'
    valid_data_name = 'sw_val.txt'
    test_data_name = 'sw_test.txt'
    # pretrained_embeddings_name = 'embeddings.pkl'
    token2idx_name = 'word2idx.json'
    label2idx_name = 'label2idx.json'
    config_name = 'config.json'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    args = {
        "epochs": 30,
        "n_batch": 64,
        "max_len": 256,
        "lr": 1e-5,
        "summary_step": 10000,
        "embedding_dim": 128,
        "hidden_size": 256,
        "n_layers": 1,
    }
    args = argparse.Namespace(**args)

    # with open(data_dir / pretrained_embeddings_name, 'rb') as f:
    #     pretrained_embeddings = pickle.load(f)

    with open(data_dir / token2idx_name, 'r') as f:
        token2idx = json.load(f)

    with open(data_dir / label2idx_name, 'r') as f:
        label2idx = json.load(f)

    with open(data_dir / config_name, 'r') as f:
        config = json.load(f)

    idx2token = {i: t for t, i in token2idx.items()}
    idx2label = {i: l for l, i in label2idx.items()}

    # 모델 생성

    learning_rate = args.lr
    batch_size = 1
    vocab_size = len(token2idx)
    embedding_dim = args.embedding_dim
    hidden_size = args.hidden_size
    output_size = len(label2idx)
    n_layers = args.n_layers
    dropout = 0.7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, n_layers, embedding_dim, device,
                           bidirectional=True)  # , weights=torch.from_numpy(pretrained_embeddings))

    model.to(device)
    loss_fn = LSR(epsilon=0.1, num_classes=output_size)

    # 학습된 모델 파라미터 불러오기

    writer = SummaryWriter(f'{save_dir}/runs')
    checkpoint_manager = CheckpointManager(save_dir)
    summary_manager = SummaryManager(save_dir)

    ckpt = checkpoint_manager.load_checkpoint('best.tar')
    model.load_state_dict(ckpt['model_state_dict'])

    test_data = pd.read_csv(data_dir / test_data_name, header=None, sep='|', names=['speaker', 'utterance', 'tag'])

    x_test, y_test = test_data['utterance'], test_data['tag']

    text_preprocess_pipeline = [sent_tokenize, stemming]  # 학습때와 동일하게 전처리
    x_test = x_test.apply(preprocess_text, processing_function_list=text_preprocess_pipeline)

    x_test = list(convert_token_to_idx(x_test, token2idx))
    y_test = list(convert_label_to_idx(y_test, label2idx))

    test_ds = DAdataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=1, collate_fn=test_ds.collate_fn)

    # 모델 평가 - ground truth 가 없을 경우에는 skip

    summ = evaluate(model, test_dl, {'loss': loss_fn, 'acc': acc}, device)

    summary_manager = SummaryManager(save_dir)
    summary_manager.load('summary.json')
    summary_manager.update(summ)
    summary_manager.save('summary.json')

    print('loss: {:3f}, acc: {:.2%}'.format(summ['loss'], summ['acc']))

    # 예측 데이터 반환

    y_temp = [0] * len(x_test)
    test_ds = DAdataset(x_test, y_temp)
    test_dl = DataLoader(test_ds, batch_size=1, collate_fn=test_ds.collate_fn, shuffle=False, drop_last=False)
    predicates = predicate(model, test_dl, device)

    y_pred = [idx2label[p] for p in predicates]
    test_data['prediction'] = y_pred
