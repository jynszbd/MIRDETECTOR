import numpy as np
from gensim.models import FastText
from gensim.models.word2vec import LineSentence


def FastTextTrain(subjectproperty,cmdpath,modelpath):
    cmd_list = []
    with open(subjectproperty,'r') as f:
        for line in f:
            id, cmd = line.strip().split(": ", 1)
            cmd_list.append(cmd)

    with open(cmdpath,'w') as wf:
        for cmd in cmd_list:
            wf.writelines(cmd)
            wf.writelines('\n')

    model = FastText(
        LineSentence(open(cmdpath, 'r', encoding='utf8')),
        vector_size=16,
        window=3,
        min_count=1,
        epochs=10,
        min_n=3,
        max_n=6,
        workers=8
    )
    model.save(modelpath)

def FastTextTest(modelpath,subjectproperty,subjectpropertyvector):
    id_to_command = {}
    model = FastText.load(modelpath)
    id_to_vector = {}
    with open(subjectproperty,'r') as f:
        for line in f:
            id, cmd = line.strip().split(": ", 1)
            id_to_command[id] = cmd

    for id in id_to_command:
        # 将命令行分解为单词
        command = id_to_command[id]
        words = command.split()
        # 获取每个单词的向量并求和
        vector_sum = [0] * model.vector_size  # 初始化为零向量
        for word in words:
            word_vector = model.wv[word]
            vector_sum = [x + y for x, y in zip(vector_sum, word_vector)]
        # 归一化向量
        vector_norm = vector_sum / np.linalg.norm(vector_sum)
        # 将归一化的向量与ID关联起来
        id_to_vector[id] = vector_norm
        # return id_to_vector
###
    with open(subjectpropertyvector,'w') as wf:
        for id in id_to_vector:
            new_line = str(id) + " " + " ".join(str(x) for x in id_to_vector[id])
            wf.writelines(new_line)
            wf.writelines('\n')


