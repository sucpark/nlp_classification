import sys
import json
import pandas as pd
from pathlib import Path
from model.utils import sent_tokenize, stemming, preprocess_text

sys.path.insert(0, '..')

if __name__ == "__main__":
    home_dir = Path('..')
    data_dir = Path('dataset')
    data_name = Path('SDAC')
    train_data_name = 'sw_train.txt'
    valid_data_name = 'sw_val.txt'
    test_data_name = 'sw_test.txt'

    train_data = pd.read_csv(home_dir / data_dir / data_name / train_data_name, header=None, sep='|',
                             names=['speaker', 'utterance', 'tag'])
    valid_data = pd.read_csv(home_dir / data_dir / data_name / valid_data_name, header=None, sep='|',
                             names=['speaker', 'utterance', 'tag'])
    test_data = pd.read_csv(home_dir / data_dir / data_name / test_data_name, header=None, sep='|',
                            names=['speaker', 'utterance', 'tag'])

    data = pd.concat([train_data['utterance'], valid_data['utterance'], test_data['utterance']], axis=0,
                     ignore_index=True)

    preprocess_pipeline = [sent_tokenize, stemming]
    data = data.apply(preprocess_text, processing_function_list=preprocess_pipeline)

    # punctuation 만 있는 sentence 존재

    # zero_sent_idx = [15546, 17228, 17851, 19150, 21495, 26890, 27815,
    #                  50593, 120070, 122050, 122228, 134316, 134342]
    #
    # data[zero_sent_idx]

    word2idx = {'<PAD>': 0}
    for sent in data:
        for word in sent:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    tags = pd.concat([train_data['tag'], valid_data['tag'], test_data['tag']], axis=0, ignore_index=True)

    labels = list(tags.unique())
    label2idx = {l: i for i, l in enumerate(labels)}

    config = {
        "n_words": len(word2idx),
        "n_tags": len(label2idx),
        "n_train": len(train_data),
        "n_valid": len(valid_data),
        "n_test": len(test_data)
    }

    with open(home_dir / data_dir / data_name / "word2idx.json", "w") as f:
        json.dump(word2idx, f)

    with open(home_dir / data_dir / data_name / "label2idx.json", "w") as f:
        json.dump(label2idx, f)

    with open(home_dir / data_dir / data_name / "config.json", "w") as f:
        json.dump(config, f)

    # embedding_dim = 100
    # pretrained_embeddings = f"glove.6b.{embedding_dim}d.txt"
    # embeddings = {}
    # with open(data_dir / pretrained_embeddings, encoding="utf8") as file:
    #         for line in file:
    #             values = line.rstrip().rsplit(' ')
    #             word = values[0]
    #             vector = np.asarray(values[1:], dtype='float32')
    #             embeddings[word] = vector

    # embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    # for word, idx in word2idx.items():
    #     if word in embeddings.keys():
    #         word_embedding = embeddings[word]
    #         embedding_matrix[idx] = word_embedding

    # with open(data_dir / "embeddings.pkl", "wb") as f:
    #     pickle.dump(embedding_matrix, f)

    # # https://team-platform.tistory.com/38