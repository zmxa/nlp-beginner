# -*- coding: utf-8 -*-
from torchtext.data import Field,TabularDataset,Iterator
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import pandas as pd
from tqdm import tqdm
import os, re
import csv
'''
    请修改WORK_PATH与3.py所在目录相同
    数据应存放在下述位置，命名相同。
    请修改GLOVE_PATH为您存放预训练向量的目录
'''
# about dataset.
FILE_TRAIN = r"data\train"
FILE_VALID = r"data\dev"
FILE_TEST = r"data\test"

# about directory.
WORK_PATH = r'E:\复旦小学的资料\nn\4'+'\\'
GLOVE_PATH = r'E:\复旦小学的资料\nn\2'+'\\'
TRAINED_VECTORS = "glove.6B.100d"

BATCH_SIZE = 10
EMBED_SIZE = 100
EMBED_SIZE_CHAR = 30
HIDDEN_SIZE = 100

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if 0, without training.
EPOCH = 5

# if True, try to load existing model in first. 
LOAD = True

def _char_tokenizer(sentence:str):
    return list(sentence)

SENTENCE = Field(sequential=True,init_token='<start>',eos_token='<end>',unk_token='<unk>',pad_token='<pad>',lower=True)
CHAR = Field(sequential=True,lower=False,tokenize=_char_tokenizer)
LABEL = Field(sequential=True,init_token='<label_start>',eos_token='<label_end>')

def load_model(path=WORK_PATH):
    file_list = list(os.walk(path))
    pattern = '(E_(\d+)_.?loss_(\d+\.\d+).+)'
    p = re.compile(pattern)
    file_list = file_list[0][2]
    best_loss = 65535.0
    temp_model=None
    num = -1
    for item in file_list:
        file = p.match(item)
        if(file):
            num = int(file.groups()[1])
            loss = file.groups()[2]
            if float(loss)<best_loss:
                temp_model=file.groups()[0]
    return num,temp_model

def dataset2iter(workpath=WORK_PATH,train_path=FILE_TRAIN,validation_path=FILE_VALID,test_path=FILE_TEST):
    fields = [('sentence',SENTENCE),('wxx',LABEL),('char',CHAR)]
    
    ######## 当你的数据行以 " 开头时，csv.reader 会认为此行包含分隔符，请额外传入以下参数。
    ######## torchtext相关文档： https://pytorch.org/text/stable/data.html#torchtext.data.TabularDataset.__init__
    ######## csv库相关文档： https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters
    csv_reader_params={'doublequote':False,'quoting':csv.QUOTE_NONE,}
    
    data_train = TabularDataset(workpath+train_path, format="tsv", fields=fields, skip_header=True,csv_reader_params=csv_reader_params)
    data_valid = TabularDataset(workpath+validation_path, format="tsv", fields=fields, skip_header=True,csv_reader_params=csv_reader_params)
    data_test = TabularDataset(workpath+test_path, format="tsv", fields=fields, skip_header=True,csv_reader_params=csv_reader_params)
    
    pretrained_vectors = Vectors(name = GLOVE_PATH+TRAINED_VECTORS+'.txt',cache=GLOVE_PATH)
    SENTENCE.build_vocab(data_train,vectors=pretrained_vectors,unk_init= lambda x:torch.nn.init.uniform_(x, a=-0.25, b=0.25))
    LABEL.build_vocab(data_train)
    CHAR.build_vocab(data_train)
    # debug
    #return data_train,data_valid,data_test
    
    iter_train = Iterator(data_train, batch_size=BATCH_SIZE,train=True,sort_key=lambda x:len(x.sentence),shuffle=True,device=DEVICE)
    iter_valid = Iterator(data_valid, batch_size=BATCH_SIZE,train=False,sort=False,shuffle=True,device=DEVICE)
    iter_test =  Iterator(data_test, batch_size=BATCH_SIZE,train=False,sort=False,shuffle=True,device=DEVICE)
    return iter_train,iter_valid,iter_test
class charEmbed(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super(charEmbed,self).__init__()
        self.drop1 = nn.Dropout(dropout_rate)
        self.embed_char = nn.Embedding(len(CHAR.vocab),padding_idx=1,
                                  embedding_dim=EMBED_SIZE_CHAR)
        nn.init.uniform_(self.embed_char.weight,-np.sqrt(3/EMBED_SIZE_CHAR),np.sqrt(3/EMBED_SIZE_CHAR))
        nn.init.uniform_(self.embed_char.weight[1],0,0)
        self.cnn = nn.Conv1d(EMBED_SIZE_CHAR, EMBED_SIZE_CHAR, kernel_size=3,padding=2)
    def forward(self,x):
        sentence = x.sentence
        batch_size = len(sentence)
        # char-embedding
        lines = []
        for item in sentence:
            temp_b=[]
            for word in item:
                if word>3:
                    temp_b.append(list(SENTENCE.vocab.itos[word]))
                else:
                    temp_b.append(['<pad>'])
            temp_b = CHAR.process(temp_b[::-1])
            temp_b = self.embed_char(temp_b).permute(1,2,0)
            line = torch.max(self.drop1(self.cnn(temp_b)),dim=2)
            lines.append(line.values.to(DEVICE))
        char_embed = torch.cat(lines[::-1]).view(batch_size,EMBED_SIZE_CHAR,len(sentence[0])).permute(0,2,1)
        return char_embed
    
class selfModel(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super(selfModel,self).__init__()
        self.label_num = len(LABEL.vocab)
        self.embed_word = nn.Embedding(len(SENTENCE.vocab),padding_idx=1,
                                  embedding_dim=EMBED_SIZE).from_pretrained(SENTENCE.vocab.vectors, freeze=True,)
        self.blstm = nn.LSTM(EMBED_SIZE+EMBED_SIZE_CHAR,HIDDEN_SIZE,bidirectional=True)
        self.fc = nn.Linear(HIDDEN_SIZE*2, len(LABEL.vocab))
        self.drop2 = nn.Dropout(dropout_rate)
        self.T = nn.init.uniform_(torch.DoubleTensor(self.label_num,self.label_num),-np.sqrt(3/self.label_num),np.sqrt(3/self.label_num)).to(DEVICE)
    def forward(self,x,char_embed):
        # enter BiLSTM
        word_embed = self.embed_word(x.sentence)
        h_embed = torch.cat((char_embed,word_embed),dim=2)
        out , _ = self.blstm(h_embed)
        out = self.fc(self.drop2(out))
        # before enter CRF
        return out
    
    def neg_caculate_sum(self,out:torch.Tensor,label:torch.Tensor):
        out = out.permute(1,0,2)
        length = self.label_num
        neg_loss = []
        out_length = len(out)
        for b in range(out_length): 
            loss = torch.DoubleTensor(self.label_num).fill_(-0.2*out_length*out_length*out_length).to(DEVICE)
            real = 0
            past = torch.DoubleTensor(self.label_num).fill_(0.).to(DEVICE)
            for single_out in out[b]:
                combine = self.T+single_out.repeat(length,1)+past.view(length,1).repeat(1,length)
                past = torch.log(torch.sum(torch.exp(combine),dim=0))
                loss+=past
            for i in range(len(label)-1):
                real+=(out[b][i][label[i][b]]+self.T[label[i][b]][label[i+1][b]])
            real+=out[b][i+1][label[i+1][b]]
            t = -real/loss.sum()
            explode_t = t.item()//5
            if explode_t >=1:
                t = t/explode_t
            neg_loss.append(-real/loss.sum())
        return sum(neg_loss)
def process(iter_inprocess,model,char_embedding,optimizer:optim.Optimizer,train=True) -> (float,float):
    if train:
        model.train()
    else :
        model.eval()
    
    total_loss = 0
    for batch in tqdm(iter_inprocess,mininterval=2):
        if train:
            optimizer.zero_grad() 
        char = char_embedding(batch)
        out = model(batch,char.to(DEVICE))
        loss = model.neg_caculate_sum(out,batch.wxx)
        if train:
            loss.backward()
            optimizer.step()
        total_loss+=loss.sum().item()
    example_size = len(iter_inprocess.dataset.examples)
    return total_loss/example_size
def main():
    iter_train,iter_valid,iter_test = dataset2iter()
    model_location = (-1,None)
    model = None
    if LOAD:
        model_location = load_model()
        if model_location[1]:
            model = torch.load(WORK_PATH+model_location[1])
    if not model:
        model = selfModel().to(DEVICE)
    char_embedding = charEmbed()
    optimizer = optim.SGD(model.parameters(),lr=0.015,weight_decay=0.05)    
    best_loss=65535.0
    v_past_loss=65535.0
    for _count in range(EPOCH):
        loss= process(iter_train,model,char_embedding,optimizer,train=True)
        print()
        print("In Train: Average loss:{:.6f}".format(loss))
        v_loss= process(iter_valid,model,char_embedding,optimizer,train=False)
        print()
        print("In Validation: Average loss:{:.6f}".format(v_loss))
        if loss<best_loss:
            best_loss = loss
            torch.save(model,WORK_PATH+'E_'+str(_count+model_location[0]+1)+'_vloss_'+str(v_loss))
        if ((v_past_loss-v_loss)/v_past_loss) < 0.001:
            break;
    print()
    print("----------Testing----------")
    t_loss= process(iter_test,model,char_embedding,optimizer,train=False)
    print()
    print("In Test: Average loss:{:.6f}".format(t_loss))
    torch.save(model,WORK_PATH+'Epoch_test_loss_'+str(t_loss))
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









    
    
    
    
    
    