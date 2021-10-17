import json, os, re
import pandas as pd
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.snippets import to_array
from keras.layers import Lambda
from keras.models import Model
from tqdm import tqdm


maxlen = 48
batch_size = 10
epochs = 20
p = 'D:/Ai/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = p +'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p +'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def read_data():

    df_train = pd.read_csv('train_20200228.csv',encoding='ANSI', dtype={'id': np.int16, 'target': np.int8})
    res = get_bert_data(df_train)
    return res

def read_dev_data():

    df_dev = pd.read_csv('dev_20200228.csv',encoding='ANSI', dtype={'id': np.int16, 'target': np.int8})
    res = get_bert_data(df_dev)
    return res    

def get_bert_data(df_data):

    length = len(df_data)
    res = []
    for i in range(length):
        label  = int(df_data['label'][i])
        query1 = clean_data(df_data['query1'][i])
        query2 = clean_data(df_data['query2'][i])
        res.append([label,query1,query2])
          
    return res

def clean_data(text):
    text = text.lower()
    text = re.sub('[?]','',text)
    text = re.sub('？','',text)
    text = re.sub('擤','擦',text)
    text = re.sub('葶苈','',text)
    text = re.sub('肟','',text)
    text = re.sub('缬','',text)
    text = re.sub('蚧','',text)
    return text

res = read_data()
train_data = []

for t in res:
    train_data.append((t[1],t[2],t[0]))  

valid = read_dev_data()

valid_data = []
for t in res:
    valid_data.append((t[1],t[2],t[0]))

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
)    

class data_generator(DataGenerator):

    def __iter__(self, random=False):

        """单条样本格式为
            输入：[CLS]两句话意思[MASK]同,text1,text2[SEP]
            输出：'相'或者'不'
        """
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []
        
        for i in idxs:

            data = self.data[i]
            text = "两句话意思相同"
            text1 = data[0]
            text2 = data[1]
            label = data[2]

            final_text = text + ':' + text1 + ',' + text2            
            token_ids, segment_ids = tokenizer.encode(final_text, maxlen=maxlen)
            
            # mask掉'相'字
            token_ids[6] = tokenizer._token_mask_id

            if label == 0:
                a_token_ids, _ = tokenizer.encode('不')
            else:
                a_token_ids, _ = tokenizer.encode('相')   

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_a_token_ids.append(a_token_ids[1:])

            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_a_token_ids = sequence_padding(
                    batch_a_token_ids, 1
                )
                yield [batch_token_ids, batch_segment_ids], batch_a_token_ids
                batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []

train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = Lambda(lambda x: x[:, 1: 2])(model.output)
model = Model(model.input, output)
model.summary()

def masked_cross_entropy(y_true, y_pred):
    """交叉熵作为loss，并mask掉padding部分的预测
    """
    y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
    y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
    return cross_entropy

model.compile(loss=masked_cross_entropy, optimizer=Adam(1e-5))

def get_ngram_set(x, n):
    """生成ngram合集，返回结果格式是:
    {(n-1)-gram: set([n-gram的第n个字集合])}
    """
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i:i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result

def predict(data):

    """
        数据格式
        text1 = data[0]
        text2 = data[1]
        label = data[2]
        方便预测和训练时候评价

    """

    text = "两句话意思相同"
    text1 = data[0]
    text2 = data[1]
    label = data[2]

    final_text = text + ':' + text1 + ',' + text2            
    token_ids, segment_ids = tokenizer.encode(final_text, maxlen=maxlen)
            
    # mask掉'相'字
    token_ids[6] = tokenizer._token_mask_id
    token_ids, segment_ids = to_array([token_ids], [segment_ids])

    # 用mlm模型预测被mask掉的部分
    probas = model.predict([token_ids, segment_ids])[0]
    res = tokenizer.decode(probas.argmax(axis=1))

    if res == '不' and label == 0:
        return '正确'
    elif res == '相' and label == 1:
        return '正确'
    elif res != '不' and res != '相':
        return '超出范围'
    else:
        return '错误'


def evaluat_vail_data(valid_data):

    right,out,all = 1,1,1

    for valid in valid_data:
        res = predict(valid)

        if res == '正确':
            right += 1
        elif res == '超出范围':
            out += 1

        all += 1      

    return right/all,out 

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        # if logs['loss'] <= self.lowest:
        #     self.lowest = logs['loss']
        #     model.save_weights('./best_model.weights')
        acc,out = evaluat_vail_data(valid_data)
        print(acc,out)
        if acc >= self.best:
            self.best = acc
            model.save_weights('./best_model.weights')

if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')