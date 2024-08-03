import torch.nn as nn
import torch.nn.functional as F
from config import *
from torchcrf import CRF
import torch
from transformers import BertModel

class ManhattanAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        # 计算查询和键的曼哈顿距离
        dist = torch.abs(q.unsqueeze(2) - k.unsqueeze(1)).sum(dim=-1)  # 展开q和k以计算曼哈顿距离

        # 由于曼哈顿距离是距离（值越小越相似），我们使用负号转换为相似度
        scores = -dist

        # 应用掩码（如果有的话）
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        # 应用softmax来计算最终的注意力权重
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        return output

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = ManhattanAttention()
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.conv = nn.Conv1d(768, 512,1)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input, mask):
        # out = self.embed(input)
        bert_out = self.bert(input, mask)[0]
        lstm_out, _ = self.lstm(bert_out)

        out = self.conv(bert_out.permute(0,2,1)).permute(0,2,1)
        out = out + lstm_out
        out = self.attention(out, out, out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input, mask)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input, mask)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # tensor.clone()会创建一个与被clone的tensor完全一样的tensor，两者不共享内存但是新tensor仍保存在计算图中，即新的tensor仍会被autograd追踪
                # 这里是在备份
                self.backup[name] = param.data.clone()
                # 归一化
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if __name__ == '__main__':
    model = Model()
    input = torch.randint(0, 3000, (100, 50))
    print(model(input, None).shape)