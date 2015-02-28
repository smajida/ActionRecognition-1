#coding:utf-8
import os

param1 = [5, 10, 15, 20]
param2 = [0.01, 0.1, 1, 10]
param3 = [0.0001, 0.001, 0.01, 0.1]

for i in param1:
	for j in param2:
		for k in param3:
			command = "echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp04/act_recog/; date; python AR_exp04.py " + str(i) + " " + str(j) + " " + str(k) + " ; date' | qsub -l"
			print command
			os.system(command)