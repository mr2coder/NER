# -*- coding:utf-8 -*-
# @author	: zjun
# @time		: 2017/07/26

import collections as col
import numpy as np

UNK = 'UNK'
VOCABU_SIZE = 10000

def readf(fname,split):
	'''
	读取文件，以数组形式返回
	param:
		fname => 文件名
		split => 每行的分隔符
	return：
		[[a,b,...],...],
		len
	'''
	with open(fname,'r') as f:
		result = [line.strip().split(split) for line in f if line!='\n']
	tag = set([x[-1] for x in result])
	tag_dict = {}
	i = 0
	for x in tag:
		tag_dict[x] = i
		i += 1
	num_class = len(tag)

	
	for x in result:
		index = tag_dict[x[-1]]
		line = [0]*num_class
		line[index] = 1
		x.append(line)
	
	return result[1:]

def load_data(data,config=None):
	'''
	生成训练数据
	param:
		data => readf函数输出结果
	'''
	word_to_index = {}
	tag_to_index = {}
	index = 0
	if not config:
		vocabu = dict(col.Counter([x[0] for x in data]).most_common(VOCABU_SIZE-1))
		vocabu = [key for key,value in vocabu.items()]
		vocabu.insert(0,UNK)
	else:
		vocabu = dict(col.Counter([x[0] for x in data]).most_common(config.vocab_size-1))
		vocabu = [key for key,value in vocabu.items()]
		vocabu.insert(0,config.UNK)
	vocabu = {value:key for key,value in enumerate(vocabu)}
	# print(vocabu)
	
	for x in data:
		if vocabu.get(x[0])==None:
			x.append(0)
		else:
			x.append(vocabu.get(x[0]))
		if tag_to_index.get(x[-3])==None:
			tag_to_index[x[-3]] = x[-2]
	result = []
	tmp_x = []
	tmp_y = []
	for x in data:
		if x[0]!='.' and x[0]!=',':
			tmp_x.append(x[-1])
			tmp_y.append(x[-2])
		else:
			result.append([tmp_x,tmp_y])
			tmp_x = []
			tmp_y = []
	result = [x for x in result if len(x[0])<37]
	return result,word_to_index,tag_to_index



def batch_iter(data, batch_size, num_epochs):
	'''
	迭代产生每个batch数据
	param:
		data => total数据集
		batch => 
		num_epochs =>
	return:
		yeild
	'''
	data = np.asarray(data)
	#获取数据长度
	data_size = len(data)
	#获得batch numbers
	batch_nums = int(len(data)/batch_size)
	for epoch in range(num_epochs):
		#随机排列数据
		shuffle_indices = np.random.permutation(np.arange(data_size))
		shuffled_data = data[shuffle_indices]
		for num_batch in range(batch_nums):
			begin_index = num_batch*batch_size
			#获取本次batch的end index，如果超过data_size，end_index=data_size
			end_index = min(begin_index+batch_size,data_size)
			#返回数据
			yield shuffled_data[begin_index:end_index]


def run():
	path = 'E:/Desktop/NeuroNER-master/data/conll2003/en/train_compatible_with_brat_bioes.txt'
	data =readf(path,' ')
	data,w_i,t_i = load_data(data)
	batch = batch_iter(data,batch_size=2,num_epochs=10)
	for x in range(1,10):
		line = batch.__next__()
		print(line)

if __name__ == '__main__':
	# run()
	data = readf('E:/Desktop/NeuroNER/data/conll2003/en/train_compatible_with_brat.txt',' ')
	# print(data[0])
	data,w_i,t_i = load_data(data)
	print(len(data))
	# print(max([len(x[0]) for x in data]))
	# dict_new = {value:key for key,value in w_i.items()}
	# max_time = 0
	# for x in data[:3]:
	# 	print(x)
		# if max_time<len(x[0]):
		# 	print(len(x[0]))
		# 	sent = [dict_new[i] for i in x[0]]
		# 	max_time = len(x[0])
		# 	print(' '.join(sent))
