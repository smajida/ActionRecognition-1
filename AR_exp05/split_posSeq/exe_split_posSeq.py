#coding:utf-8
import os

param1 = [20, 50, 100, 150, 200]
param2 = [10, 20, 30, 40, 50]

for i in param1:
	for j in param2:
		cmd = "echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp05/split_posSeq/; date; python split_posSeq.py " + str(i) + " " + str(j) + " ; date' | qsub"
		# print cmd
		os.system(cmd)