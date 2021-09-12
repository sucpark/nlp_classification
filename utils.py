import nltk


def sent2words(sent):
    temp_words = nltk.word_tokenize(sent)
    temp_words = [w.lower() for w in temp_words]  # 추가 전처리 가능 (punctation, stopwords, .. )
    
    return temp_words

def add_padding(token_ls, max_len):
    """
    # Sequence Length를 맞추기 위한 padding
    """
    pad = '<PAD>'
    seq_length_ls = []
    
    for i, tokens in enumerate(token_ls):
        seq_length = len(tokens)

        if seq_length < max_len:
            seq_length_ls.append(seq_length)
            token_ls[i] += [pad] * (max_len - seq_length)

        elif seq_length >= max_len:
            seq_length_ls.append(max_len)
            token_ls[i] = tokens[:max_len]
            
    return token_ls, seq_length_ls

def convert_token_to_idx(tokens, token2idx):
    for token in tokens:
        yield [token2idx[t] for t in token]
    return

def convert_label_to_idx(labels, label2idx):
    temp = []
    for label in labels:
        temp.append(label2idx[label])
    return temp

def evaluate(model, data_loader, metrics, device):
    
    loss = 0
    if model.training:
        model.eval()
        
    summary = {metric: 0 for metric in metrics}
        
    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)
        
        with torch.no_grad():
            y_hat_mb = model(x_mb)
            
            for metric in metrics:
                summary[metric] += metrics[metric](y_hat_mb, y_mb).item()*y_mb.size()[0]
    else:
        for metric in metrics:
            summary[metric] /= len(data_loader.dataset)
    
    return summary

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
