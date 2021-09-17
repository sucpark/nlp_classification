import argparse
import json
import pandas as pd
from pathlib import Path
from model.utils import sent_tokenize, stemming, preprocess_text

parser = argparse.ArgumentParser(description="Evaluating nlp classification model")
parser.add_argument('--data_dir', default='dataset', help="Directory containing data")
parser.add_argument('--data_name', default='SDAC')

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_name = Path(args.data_name)
    train_data_name = 'sw_train.csv'
    valid_data_name = 'sw_val.csv'
    test_data_name = 'sw_test.csv'

    train_data = pd.read_csv(data_dir / data_name / train_data_name, header=None, sep=',',
                             names=['speaker', 'utterance', 'tag'])
    valid_data = pd.read_csv(data_dir / data_name / valid_data_name, header=None, sep=',',
                             names=['speaker', 'utterance', 'tag'])
    test_data = pd.read_csv(data_dir / data_name / test_data_name, header=None, sep=',',
                            names=['speaker', 'utterance'])

    data = pd.concat([train_data['utterance'], valid_data['utterance'], test_data['utterance']], axis=0,
                     ignore_index=True)

    preprocess_pipeline = [sent_tokenize, stemming]
    data = data.apply(preprocess_text, processing_function_list=preprocess_pipeline)

    word2idx = {'<PAD>': 0}
    for sent in data:
        for word in sent:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    tags = pd.concat([train_data['tag'], valid_data['tag']], axis=0, ignore_index=True)
    labels = list(tags.unique())
    label2idx = {l: i for i, l in enumerate(labels)}

    config = {
        "n_words": len(word2idx),
        "n_labels": len(label2idx),
        "n_train": len(train_data),
        "n_valid": len(valid_data),
        "n_test": len(test_data)
    }

    with open(data_dir / data_name / "word2idx.json", "w") as f:
        json.dump(word2idx, f)

    with open(data_dir / data_name / "label2idx.json", "w") as f:
        json.dump(label2idx, f)

    with open(data_dir / data_name / "config.json", "w") as f:
        json.dump(config, f)
