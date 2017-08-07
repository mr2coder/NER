# -*- coding:utf-8 -*-
# @author	: zjun
# @time		: 2017/06/26

import tensorflow as tf
import numpy as np
import lstm
from lstm import config as conf
import importdata as dataset
from time import strftime,gmtime
import time

#配置参数
CONFIG = {
	'hidden_size':30,
	'embedding_size':150,
	'batch_size':30,
	'dropout_rate':0.4,
	'layes':6,
	'num_epochs':1000,
	'vocab_size':8000,
	'number_of_classes':9,#17
	'sentence_max_len':36,
	'gradient_clipping_value':5,
	'learning_rate':0.005,
	'optimizer':'adadelta',
	'UNK':'UNK'
}

def run():
	'''
	主函数
	'''
	#训练数据文件
	fname = 'E:/Desktop/NeuroNER/data/conll2003/en/train_compatible_with_brat.txt'
	split = ' '
	data = dataset.readf(fname,split)
	config = conf(CONFIG)

	train_input,word_to_index,tag_to_index = dataset.load_data(data,config=config)

	# train_input = train_input[:100]
	batch_iter = dataset.batch_iter(train_input,config.batch_size,config.num_epochs)

	#每个epoch所进行batch的次数
	batch_nums = len(train_input)//config.batch_size

	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			#模型初始化
			model = lstm.BiLSTM_CRF(config)

			#tensorboard信息保存地址
			log_dir = 'E:/Desktop/NeuroNER/contrib/log'
			merged = tf.summary.merge_all()  
			train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

			#参数初始化
			sess.run(tf.global_variables_initializer())
			
			for epoch in range(config.num_epochs*batch_nums):
				#获取当前batch的训练数据
				batch_set = batch_iter.__next__()
				batch_x = [x[0] for x in batch_set]
				batch_y = [x[1] for x in batch_set]
				length = [len(x[0]) for x in batch_set]
				#补齐batch_x
				for x in batch_x:
					x.extend([0]*(config.sentence_max_len-len(x)))
				for x in batch_y:
					x.extend([[0]*config.number_of_classes]*(config.sentence_max_len-len(x)))
				#运行一次
				# print(np.array(batch_x).shape)
				summary,acc,loss,_ = sess.run([merged,model.accuracy,model.loss,model.train_op],
					feed_dict={
					model.input_words:batch_x,
					model.input_labels:batch_y,
					model.sequence_length:length,
					model.dropout_rate:config.dropout_rate
					})

				#每次训练后写入文件
				train_writer.add_summary(summary, epoch)
				#打印输出训练信息
				if epoch%10==0:
					print('{0} epochs:{1},steps:{2},accuracy:{3},loss:{4}'.format(
						time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
						epoch//batch_nums,epoch,acc,loss))



if __name__ == '__main__':
	run()