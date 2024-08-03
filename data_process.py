import pandas as pd
from config import *

# 生成标签表
def generate_label():
    # 字和标签之间用空格分隔
    df = pd.read_csv(TRAIN_SAMPLE_PATH, delimiter=' ',usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().tolist()
    # seqeval要求标签分隔符只能是中划线
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=None, index=None)

if __name__ == '__main__':
    # 生成标签表，生成后顺序是乱的，有可能B和I不在一起，可以适当手动调整。
    generate_label()
