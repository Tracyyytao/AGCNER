import os

BASE_PATH = os.path.dirname(__file__)

TRAIN_SAMPLE_PATH = os.path.join(BASE_PATH, './input/agcner_ner/train.txt')
TEST_SAMPLE_PATH = os.path.join(BASE_PATH, './input/agcner_ner/test.txt')
DEV_SAMPLE_PATH = os.path.join(BASE_PATH, './input/agcner_ner/dev.txt')


LABEL_PATH = os.path.join(BASE_PATH, './output/label.txt')

TARGET_SIZE = 27  # 标签数量

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 3000

HIDDEN_SIZE = 256

LR = 5e-5
EPOCH = 30


MODEL_DIR = os.path.join(BASE_PATH, './output/model/')

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# bert改造
BERT_MODEL = './huggingface/chinese-roberta-wwm-ext'
ERNIE_MODEL = './huggingface/ernie-3.0-base-zh'
EMBEDDING_DIM = 768
MAX_POSITION_EMBEDDINGS = 512