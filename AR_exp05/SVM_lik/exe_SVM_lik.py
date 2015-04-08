#coding: utf-8
import os

for s in [3, 4]:
	for K in [50, 100, 150]:
		for T in [10, 30, 50]:
			for C in [0.01]:
				for gamma in [0.01, 0.1]:
					for lenSeq in [10, 20]:
						cmd = "echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp05/SVM_lik/; date; python SVM_lik.py %d %d %d %f %f %d; date' | qsub" % (s,K,T,C,gamma,lenSeq)
						os.system(cmd)