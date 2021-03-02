# -*- coding: utf-8 -*-
from torchtext.data import Field,TabularDataset,BucketIterator,Iterator
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import os, re
'''
    数据应分别放置在test与train文件夹下，分别命名test.tsv与train.tsv
    请修改WORK_PATH与2.py存放目录相同，不必修改DATA_PATH
'''
DATA_PATH = r"train\train.tsv"
WORK_PATH = r'E:\复旦小学的资料\nn\2'
TRAINED_VECTORS = "glove.6B.100d"

BATCH_SIZE = 10
EMBED_SIZE = 100
HIDDEN_SIZE = 32

EPOCH = 10
LOAD = False
TYPE = 0

TEXT = Field(sequential=(True),use_vocab=(True),pad_token='<pad>',unk_token='<unk>',lower=True)
LABEL = Field(sequential=(False),batch_first=(True),pad_token=None,unk_token=None)
    
def dataset2iter(workpath=WORK_PATH,data_path=DATA_PATH):
    fields = [("PhraseId", None),("SentenceId", None),('Phrase', TEXT),('Sentiment', LABEL)]
    
    data_all = TabularDataset(path=WORK_PATH+'\\'+DATA_PATH, format='tsv',fields=fields)
    data_train, data_valid, data_test = data_all.split( split_ratio=[0.6,0.2,0.2])
    
    pretrained_vectors = Vectors(name = WORK_PATH+'\\'+TRAINED_VECTORS+'.txt',cache=WORK_PATH)
    TEXT.build_vocab(data_train,vectors=pretrained_vectors )
    LABEL.build_vocab(data_train)
    
    iter_train = BucketIterator(data_train, batch_size=BATCH_SIZE,sort_key=lambda x:len(x.Phrase),sort=True)
    iter_valid = BucketIterator(data_valid, batch_size=BATCH_SIZE,train=False,sort_key=lambda x:len(x.Phrase))
    iter_test = BucketIterator(data_test, batch_size=BATCH_SIZE,train=False,sort_key=lambda x:len(x.Phrase))
    return iter_train,iter_valid,iter_test

class selfRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,dropout_rate=0.5):
        super(selfRNN,self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        
        self.embed=nn.Embedding(len(TEXT.vocab), embed_size, padding_idx=1).from_pretrained(TEXT.vocab.vectors,freeze=True)
        self.layer1=nn.LSTM(embed_size, hidden_size,bidirectional=True)
        self.dropout=nn.Dropout(dropout_rate)
        self.layer2=nn.Linear(hidden_size*2, output_size)
    
    def forward(self, x, batch_size):
        h0 = torch.zeros(2,batch_size,self.hidden_size)
        c0 = torch.zeros(2,batch_size,self.hidden_size)
        
        embed_input = self.embed(x)
        out,_ = self.layer1(embed_input,(h0,c0))
        out = self.dropout(out)
        out = self.layer2(out[-1,:,:])
        out = torch.squeeze(out,dim=0)
        return out
    
class selfCNN(nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,dropout_rate=0.5):
        super(selfCNN,self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        
        self.embed=nn.Embedding(len(TEXT.vocab), embed_size, padding_idx=1).from_pretrained(TEXT.vocab.vectors,freeze=True)
        
        #In oreder to fix the problem of seq_len, use conv2d rather than conv1d.
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=hidden_size, kernel_size=(3, embed_size), padding=(1,0))
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=hidden_size, kernel_size=(5, embed_size), padding=(2,0))
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(2*hidden_size, output_size)
    def forward(self,x,batch_size):
        x=self.embed(x).unsqueeze(1)
        x=x.permute(2,1,0,3)
        x1 = F.relu(self.conv1(x)).squeeze(3)
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        
        x2 = F.relu(self.conv2(x)).squeeze(3)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        out = torch.cat((x1, x2), dim=1)
        out = self.dropout(out)
        out = self.layer2(out)
        return out
def process(iter_inprocess,model,optimizer:optim.Optimizer,lossfunc,train=True):
    if train:
        model.train()
    else :
        model.eval()
    
    step = 0
    total_loss = 0
    total_correct = 0
    for batch in tqdm(iter_inprocess):
        
        step+=1
        if train:
            optimizer.zero_grad() 
        batch_text=(batch.Phrase)
        batch_label=batch.Sentiment
        
        out = model(batch_text,batch.batch_size)
        loss = lossfunc(out,batch_label)
        if train:
            loss.backward()
            optimizer.step()
        
        correct = (torch.max(out, dim=1)[1].view(batch_label.size()) == batch_label).sum()
        total_correct+=correct.item()
        total_loss+=loss.item()
#        if (not (step%100)) and train:
#            print("Average loss:{:.6f}; Accuracy:{:.6f}".format(total_loss/step,total_correct/len(iter_inprocess.dataset)))
#            print()
    return total_loss/step,total_correct/len(iter_inprocess.dataset)
            
def load_model(path=WORK_PATH):
    file_list = list(os.walk(path))
    pattern = '(Epoch_(\d+)_loss_(\d+\.\d+))'
    p = re.compile(pattern)
    file_list = file_list[0][2]
    best_loss = 65535.0
    temp_model=None
    for item in file_list:
        file = p.match(item)
        if(file):
            num = file.groups()[1]
            loss = file.groups()[2]
            if float(loss)<best_loss:
                temp_model=file.groups()[0]
    return num,temp_model
            
def main():
    model_list = [selfCNN,selfRNN]
    iter_train,iter_valid,iter_test = dataset2iter()
    model=None
    if LOAD:
        model_location = load_model()
        if model_location:
            model = torch.load(WORK_PATH+'\\'+model_location[1])
    if not model:
        model = model_list[TYPE](EMBED_SIZE,HIDDEN_SIZE,6)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    
    print("----------Training----------")
    best_loss=65535.0
    v_past_loss=65535.0
    for _count in range(EPOCH):
        loss,correct = process(iter_train,model,optimizer,loss_function,train=True)
        print("After Test: Average loss:{:.6f}; Accuracy:{:.6f}".format(loss,correct))
        v_loss,v_correct = process(iter_valid,model,optimizer,loss_function,train=False)
        print("In Validation: Average loss:{:.6f}; Accuracy:{:.6f}".format(v_loss,v_correct))
        if loss<best_loss:
            best_loss = loss
            torch.save(model,WORK_PATH+'\\Epoch_'+str(_count)+'_loss_'+str(loss))
        if ((v_past_loss-v_loss)/v_past_loss) < 0.001:
            break;

    print("----------Testing----------")
    t_loss,t_correct = process(iter_test,model,optimizer,loss_function,train=False)
    print("In Test: Average loss:{:.6f}; Accuracy:{:.6f}".format(t_loss,t_correct))
    torch.save(model,WORK_PATH+'\\Epoch_test_loss_'+str(t_loss))
    
   

if __name__ == '__main__':
    main()
    
    
    
    
        
    























