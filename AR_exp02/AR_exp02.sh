#!/bin/sh
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp02/; date; echo 20|python AR_exp02.py; date' | qsub -l nodes=4
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp02/; date; echo 50|python AR_exp02.py; date' | qsub -l nodes=4
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp02/; date; echo 100|python AR_exp02.py; date' | qsub -l nodes=4
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp02/; date; echo 150|python AR_exp02.py; date' | qsub -l nodes=4
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/AR_exp02/; date; echo 200|python AR_exp02.py; date' | qsub -l nodes=4