#coding: utf-8
import os

for lenSeq_hmm in [5, 20, 50]:
	for n_states in [5, 10, 15, 20]:
		for lenSeq_svm in [5, 20, 50]:
			cmd = "echo 'source .bashrc; cd /home/kodama-y/work/ActionRecognition/AR_exp06/; date; \
				   python SVM_lik.py %d %d %d; date' | qsub" % (lenSeq_hmm, n_states, lenSeq_svm)
			os.system(cmd)