# -*- coding: utf-8 -*-
from torchtext.data import Field,TabularDataset,Iterator
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import pandas as pd
from tqdm import tqdm
import os, re#, time
'''
    请修改WORK_PATH与3.py所在目录相同
    数据应存放在下述位置，命名相同。
    请修改GLOVE_PATH为您存放预训练向量的目录
'''
# about dataset.
FILE_TRAIN = r"snli_1.0\snli_1.0_train.txt"
FILE_VALID = r"snli_1.0\snli_1.0_dev.txt"
FILE_TEST = r"snli_1.0\snli_1.0_test.txt"

# about directory.
WORK_PATH = r'E:\复旦小学的资料\nn\3'+'\\'
GLOVE_PATH = r'E:\复旦小学的资料\nn\2'+'\\'
TRAINED_VECTORS = "glove.6B.100d"

BATCH_SIZE = 100
EMBED_SIZE = 100
HIDDEN_SIZE = 100

# if 0, without training.
EPOCH = 0

# if True, try to load existing model in first. 
LOAD = True

# default type when creating model. 
# 0 -->  LSTM with attention
# 1 -->  LSTM with word-by-word attention
# 2 -->  two-way LSTM with word-by-word attention
TYPE = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use for
def load_model(path=WORK_PATH):
    file_list = list(os.walk(path))
    pattern = '(E_(\d+)_?T_[\d]+_.loss_(\d+\.\d+))'
    p = re.compile(pattern)
    file_list = file_list[0][2]
    best_loss = 65535.0
    temp_model=None
    num = -1
    for item in file_list:
        file = p.match(item)
        if(file):
            num = file.groups()[1]
            loss = file.groups()[2]
            if float(loss)<best_loss:
                temp_model=file.groups()[0]
    return num,temp_model

#Data process
TKNIZER_PATTERN = re.compile("[^\w]+")

LABEL = Field(sequential=(False),batch_first=(True),pad_token=None,unk_token=None)
SENTENCE_FIRST = Field(sequential=(True),tokenize=lambda x: TKNIZER_PATTERN.split(x)[1:-1],lower=True,unk_token='<unk>')
SENTENCE_SECOND = Field(sequential=(True),tokenize=lambda x: TKNIZER_PATTERN.split(x)[1:-1],lower=True,unk_token='<unk>',init_token='<start>')

def dataset2iter(workpath=WORK_PATH,train_path=FILE_TRAIN,validation_path=FILE_VALID,test_path=FILE_TEST):
    fields =[('gold_label',LABEL),
         ('sentence1_binary_parse',SENTENCE_FIRST),
         ('sentence2_binary_parse',SENTENCE_SECOND),
         ]

    data_train = TabularDataset(workpath+train_path, format="tsv", fields=fields, skip_header=True)
    data_valid = TabularDataset(workpath+validation_path, format="tsv", fields=fields, skip_header=True)
    data_test = TabularDataset(workpath+test_path, format="tsv", fields=fields, skip_header=True)

    pretrained_vectors = Vectors(name = GLOVE_PATH+TRAINED_VECTORS+'.txt',cache=GLOVE_PATH)
    SENTENCE_FIRST.build_vocab(data_train,vectors=pretrained_vectors,unk_init= lambda x:torch.nn.init.uniform_(x, a=-0.25, b=0.25) )
    SENTENCE_SECOND.build_vocab(data_train,vectors=pretrained_vectors,unk_init= lambda x:torch.nn.init.uniform_(x, a=-0.25, b=0.25) )
    LABEL.build_vocab(data_train)

    iter_train = Iterator(data_train, batch_size=BATCH_SIZE,train=True,sort_key=lambda x:len(x.sentence1_binary_parse),shuffle=True,device=DEVICE)
    iter_valid = Iterator(data_valid, batch_size=BATCH_SIZE,train=False,sort=False,shuffle=True,device=DEVICE)
    iter_test =  Iterator(data_test, batch_size=BATCH_SIZE,train=False,sort=False,shuffle=True,device=DEVICE)
    return iter_train,iter_valid,iter_test

class selfLSTM(nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,dropout_rate=0.5):
        super(selfLSTM,self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size

        self.embed = nn.Embedding(len(SENTENCE_SECOND.vocab),padding_idx=1,
                                  embedding_dim=EMBED_SIZE).from_pretrained(SENTENCE_SECOND.vocab.vectors, freeze=True,)
        self.lstm_left = nn.LSTM(input_size=embed_size,hidden_size=hidden_size)
        self.lstm_right = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,)
        
        k=self.hidden_size
        self.linear_y = nn.Linear(k,k,bias=False)
        self.linear_h = nn.Linear(k,k,bias=False)
        self.linear_t = nn.Linear(k,1,bias=False)
        self.linear_x = nn.Linear(k,k,bias=False)
        self.linear_p = nn.Linear(k,k,bias=False)
        self.dropout=nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(k,output_size)
    def forward(self,x):
        x_s1 =x.sentence1_binary_parse
        x_s2 =x.sentence2_binary_parse
        L = len(x_s1)
        embed_x_s1 = self.embed(x_s1)
        Y,(h_l,c_l) = self.lstm_left(embed_x_s1)
        Y_Wy = self.linear_y(Y).permute(1,2,0)
        
        embed_x_s2 = self.embed(x_s2)
        _ , (h_n, c_n) = self.lstm_right(embed_x_s2,(h_l,c_l))
        h_Wh = self.linear_h(h_n)
        h_Wh = h_Wh.squeeze().unsqueeze(2)
        e_l = torch.ones(L).unsqueeze(0)
        e_l = e_l.to(DEVICE)
        h_outproduct = torch.matmul(h_Wh,e_l)
        M = torch.tanh(h_outproduct+Y_Wy)
        
        alpha = F.softmax(self.linear_t(M.permute(0,2,1)),dim=1)
        r = torch.matmul(Y.permute(1,2,0),alpha)
        h_out = torch.tanh(self.linear_p(r.squeeze())+self.linear_x(h_n.squeeze()))
        out = self.output_layer(self.dropout(h_out))
        return out
    
class selfLSTM_2(nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,dropout_rate=0.5):
        super(selfLSTM_2,self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size

        self.embed = nn.Embedding(len(SENTENCE_SECOND.vocab),padding_idx=1,
                                  embedding_dim=EMBED_SIZE).from_pretrained(SENTENCE_SECOND.vocab.vectors, freeze=True,)
        self.lstm_left = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,)
        self.lstm_right = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,)
        
        k=self.hidden_size
        self.linear_y = nn.Linear(k,k,bias=False)
        self.linear_h = nn.Linear(k,k,bias=False)
        self.linear_t = nn.Linear(k,1,bias=False)
        self.linear_x = nn.Linear(k,k,bias=False)
        self.linear_p = nn.Linear(k,k,bias=False)
        self.linear_r = nn.Linear(k,k,bias=False)
        self.dropout=nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(k,output_size)
        
    def forward(self,x):
        x_s1 =x.sentence1_binary_parse
        x_s2 =x.sentence2_binary_parse
        L = len(x_s1)
        embed_x_s1 = self.embed(x_s1)
        Y,(h_l,c_l) = self.lstm_left(embed_x_s1)
        Y_Wy = self.linear_y(Y).permute(1,2,0)
        
        embed_x_s2 = self.embed(x_s2)
        Y_r , (h_n, c_n) = self.lstm_right(embed_x_s2,(h_l,c_l))
        
        r = torch.zeros(self.hidden_size).to(DEVICE)
        e_l = torch.ones(L).unsqueeze(0)
        e_l = e_l.to(DEVICE)
        
        for t in range(len(x_s2)):
            
            h_Wh = self.linear_h(Y_r[t])+self.linear_r(r)
            h_Wh = h_Wh.squeeze().unsqueeze(2)
            h_outproduct = torch.matmul(h_Wh,e_l)
            M = torch.tanh(h_outproduct+Y_Wy)
            alpha = F.softmax(self.linear_t(M.permute(0,2,1)),dim=1)
            r = torch.matmul(Y.permute(1,2,0),alpha)
            r = r.squeeze()
        h_out = F.tanh(self.linear_p(r.squeeze())+self.linear_x(h_n.squeeze()))
        out = self.output_layer(self.dropout(h_out))
        return out
    
class selfLSTM_3(nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,dropout_rate=0.5):
        super(selfLSTM_3,self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size

        self.embed = nn.Embedding(len(SENTENCE_SECOND.vocab),padding_idx=1,
                                  embedding_dim=EMBED_SIZE).from_pretrained(SENTENCE_SECOND.vocab.vectors, freeze=True,)
        self.lstm_left1 = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,)
        self.lstm_right1 = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,)
        
        k=self.hidden_size
        self.linear_y1 = nn.Linear(k,k,bias=False)
        self.linear_h1 = nn.Linear(k,k,bias=False)
        self.linear_t1 = nn.Linear(k,1,bias=False)
        self.linear_x1 = nn.Linear(k,k,bias=False)
        self.linear_p1 = nn.Linear(k,k,bias=False)
        self.linear_r1 = nn.Linear(k,k,bias=False)
        self.dropout1=nn.Dropout(dropout_rate)
        
        self.lstm_left2 = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,)
        self.lstm_right2 = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,)
        self.linear_y2 = nn.Linear(k,k,bias=False)
        self.linear_h2 = nn.Linear(k,k,bias=False)
        self.linear_t2 = nn.Linear(k,1,bias=False)
        self.linear_x2 = nn.Linear(k,k,bias=False)
        self.linear_p2 = nn.Linear(k,k,bias=False)
        self.linear_r2 = nn.Linear(k,k,bias=False)

        self.dropout2=nn.Dropout(dropout_rate)
        
        self.combined_out = nn.Linear(2*k,output_size)
    def forward(self,x):
        x_s1 =x.sentence1_binary_parse
        x_s2 =x.sentence2_binary_parse
        L = len(x_s1)
        embed_x_s1 = self.embed(x_s1)
        embed_x_s2 = self.embed(x_s2)
        
        #1
        Y1,(h_l1,c_l1) = self.lstm_left1(embed_x_s1)
        Y_Wy1 = self.linear_y1(Y1).permute(1,2,0)
        
        Y_r1 , (h_n1, c_n1) = self.lstm_right1(embed_x_s2,(h_l1,c_l1))
        r1 = torch.zeros(self.hidden_size).to(DEVICE)
        e_l1 = torch.ones(L).unsqueeze(0)
        e_l1 = e_l1.to(DEVICE)
        
        for t in range(len(x_s2)):
            
            h_Wh1 = self.linear_h1(Y_r1[t])+self.linear_r1(r1)
            h_Wh1 = h_Wh1.squeeze().unsqueeze(2)
            h_outproduct1 = torch.matmul(h_Wh1,e_l1)
            M1 = F.tanh(h_outproduct1+Y_Wy1)
            alpha1 = F.softmax(self.linear_t1(M1.permute(0,2,1)),dim=1)
            r1 = torch.matmul(Y1.permute(1,2,0),alpha1)
            r1 = r1.squeeze()
        h_out1 = F.tanh(self.linear_p1(r1.squeeze())+self.linear_x1(h_n1.squeeze()))
        out1 = self.dropout1(h_out1)
        
        #2
        Y2,(h_l2,c_l2) = self.lstm_left2(embed_x_s2)
        Y_Wy2 = self.linear_y2(Y2).permute(1,2,0)
        
        Y_r2 , (h_n2, c_n2) = self.lstm_right2(embed_x_s1,(h_l2,c_l2))
        L = len(x_s2)
        r2 = torch.zeros(self.hidden_size).to(DEVICE)
        e_l2 = torch.ones(L).unsqueeze(0)
        e_l2 = e_l2.to(DEVICE)
        
        for t in range(len(x_s1)):
            
            h_Wh2 = self.linear_h2(Y_r2[t])+self.linear_r2(r2)
            h_Wh2 = h_Wh2.squeeze().unsqueeze(2)
            h_outproduct2 = torch.matmul(h_Wh2,e_l2)
            M2 = F.tanh(h_outproduct2+Y_Wy2)
            alpha2 = F.softmax(self.linear_t2(M2.permute(0,2,1)),dim=1)
            r2 = torch.matmul(Y2.permute(1,2,0),alpha2)
            r2 = r2.squeeze()
        h_out2 = torch.tanh(self.linear_p2(r2.squeeze())+self.linear_x2(h_n2.squeeze()))
        out2 = self.dropout2(h_out2)
        out = torch.cat((out1,out2),dim=1)
        out = self.combined_out(out)
        return out
# use in train
def process(iter_inprocess,model,optimizer:optim.Optimizer,lossfunc,train=True) -> (float,float):
    if train:
        model.train()
    else :
        model.eval()
    
    total_loss = 0
    total_correct = 0
    
    
    for batch in tqdm(iter_inprocess,mininterval=2):
        if train:
            optimizer.zero_grad() 
        
        out = model(batch)
        loss = lossfunc(out,batch.gold_label)
        if train:
            loss.backward()
            optimizer.step()
        
        correct = (torch.max(out, dim=1)[1].view(batch.gold_label.size()) == batch.gold_label ).sum()
        total_correct+=correct.item()
        total_loss+=loss.item()
    example_size = len(iter_inprocess.dataset.examples)
    return total_loss/example_size,total_correct/len(iter_inprocess.dataset)

def main():
    
    model_list = [selfLSTM,selfLSTM_2,selfLSTM_3]
    model = None
    iter_train,iter_valid,iter_test = dataset2iter()
    model_location = (-1,None)
    if LOAD:
        model_location = load_model()
        if model_location[1]:
            model = torch.load(WORK_PATH+'\\'+model_location[1])
    if not model:
        model = model_list[TYPE](EMBED_SIZE,HIDDEN_SIZE,len(LABEL.vocab),).to(DEVICE)
        
    optimizer = optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999))
    loss_function = nn.CrossEntropyLoss().to(DEVICE)
    
    best_loss=65535.0
    v_past_loss=65535.0
    for _count in range(EPOCH):
        loss,correct = process(iter_train,model,optimizer,loss_function,train=True)
        print()
        print("In Train: Average loss:{:.6f}; Accuracy:{:.6f}".format(loss,correct))
        v_loss,v_correct = process(iter_valid,model,optimizer,loss_function,train=False)
        print()
        print("In Validation: Average loss:{:.6f}; Accuracy:{:.6f}".format(v_loss,v_correct))
        if loss<best_loss:
            best_loss = loss
            torch.save(model,WORK_PATH+'\\E_'+str(_count+model_location[0]+1)+'_T_'+str(TYPE)+'_vloss_'+str(v_loss))
        if ((v_past_loss-v_loss)/v_past_loss) < 0.001:
            break;
    print()
    print("----------Testing----------")
    t_loss,t_correct = process(iter_test,model,optimizer,loss_function,train=False)
    print()
    print("In Test: Average loss:{:.6f}; Accuracy:{:.6f}".format(t_loss,t_correct))
    torch.save(model,WORK_PATH+'\\Epoch_test_loss_'+str(t_loss))

if __name__ == '__main__':
    main()

# killed by user.
# 
# Epoch passed: 3
# 100%|██████████| 5502/5502 [30:31<00:00,  3.00it/s]
# In Train: Average loss:0.007749; Accuracy:0.662919
# 100%|██████████| 100/100 [00:24<00:00,  4.15it/s]
# In Validation: Average loss:0.008477; Accuracy:0.658500
#
# use that model's parameter:
# 100%|██████████| 100/100 [00:07<00:00, 13.23it/s]
# In Test: Average loss:0.008632; Accuracy:0.658000


