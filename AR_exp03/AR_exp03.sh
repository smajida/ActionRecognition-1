#!/bin/sh
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp03/; date; echo 1|python AR_exp03.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp03/; date; echo 2|python AR_exp03.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp03/; date; echo 5|python AR_exp03.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp03/; date; echo 10|python AR_exp03.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp03/; date; echo 15|python AR_exp03.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp03/; date; echo 20|python AR_exp03.py; date' | qsub -l nodes=2