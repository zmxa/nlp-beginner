import os

def data_process(path=os.getcwd()):
    file_list = list(os.walk(path))
    temp_write_list = []
    for file in file_list[0][2]:
        if file.endswith(r'.txt'):
            new_file = [file.split('.')[0],]
            with open(path+'\\'+file) as fp:
                sentence=''
                label=''
                
                t = fp.readline()
                while(t!=''):
                    if t=='\r\n' or t=='\n':
                        new_file.append((sentence,label))
                        sentence=''
                        label=''
                    else:
                        tt = t.split()
                        sentence += (tt[0]+' ')
                        label += (tt[-1]+' ')
                    t = fp.readline()
            temp_write_list.append(new_file)
            new_file = []
    for item in temp_write_list:
        name = path+'\\'+item[0]+''
        item = item[1:]
        with open(name,'w') as fp:
            fp.write('sentence\twxx\tchar\n')
            for pair in item:
                if(pair[0]=='' or pair[1]==''):
                    continue
                fp.write(pair[0]+'\t'+pair[1]+'\t'+pair[0]+'\n')
    return temp_write_list








if __name__ == '__main__':
    a = data_process()
