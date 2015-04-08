#coding:utf-8
import os

param1 = [50, 100, 150]
param2 = [10, 30, 50]

for i in param1:
	for j in param2:
		cmd = "echo 'source .bashrc; cd /home/kodama-y/work/ActionRecognition/AR_exp05/split_posSeq/; date; python split_posSeq.py " + str(i) + " " + str(j) + " ; date' | qsub"
		# print cmd
		os.system(cmd)