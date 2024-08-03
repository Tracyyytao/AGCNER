import torch
from torch.utils import data
from config import *
import pandas as pd
from seqeval.metrics import classification_report

from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score

from transformers import BertTokenizer
from transformers import logging

logging.set_verbosity_warning()


def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)


class Dataset(data.Dataset):

    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'test':
            sample_path = TEST_SAMPLE_PATH
        self.sample_data = self.parse_sample(sample_path)
        _, self.label2id = get_label()
        # 初始化Bert
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def parse_sample(self, file_path):
        with open(file_path, encoding='utf-8') as file:
            content = file.read()
            result = []
            arr = content.strip().split('\n\n')
            for item in arr:
                text = []
                label = []
                for line in item.split('\n'):
                    t, l = line.split(' ')
                    text.append(t)
                    label.append(l)
                result.append((text, label))
        return result

    def __len__(self):
        return len(self.sample_data)

    def __getitem__(self, index):
        text, label = self.sample_data[index]
        label_o_id = self.label2id['O']
        # input = [self.word2id.get(w, word_unk_id) for w in df['word']]
        # 注意：先自己将句子做分词，再转id，避免bert自动分词导致句子长度变化
        input = self.tokenizer.encode(text, add_special_tokens=False)
        target = [self.label2id.get(l.replace('_', '-'), label_o_id) for l in label]
        # return input, target
        # bert要求句子长度不能超过512
        return input[:MAX_POSITION_EMBEDDINGS], target[:MAX_POSITION_EMBEDDINGS]


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = len(batch[0][0])
    input = []
    target = []
    mask = []
    for item in batch:
        pad_len = max_len - len(item[0])
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        mask.append([1] * len(item[0]) + [0] * pad_len)
    return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool()


def extract(label, text):
    i = 0
    res = []
    while i < len(label):
        if label[i] != 'O':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and label[i] == 'I-' + name:
                end = i
                i += 1
            res.append([name, text[start:end + 1]])
        else:
            i += 1
    return res


#def report(y_true, y_pred):
    #return classification_report(y_true, y_pred)
def getMetrics(pre_labels, true_labels):
    precision = precision_score(true_labels, pre_labels)
    recall = recall_score(true_labels, pre_labels)
    f1 = f1_score(true_labels, pre_labels)
    print(classification_report(true_labels, pre_labels, digits=4))
    print("precision,recall,f1:", precision, recall, f1)
    return precision, recall, f1


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=100, collate_fn=collate_fn)
    print(iter(loader).next())