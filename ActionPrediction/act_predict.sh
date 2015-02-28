#!/bin/sh
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/act_predict/; date; echo 1|python act_predict.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/act_predict/; date; echo 2|python act_predict.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/act_predict/; date; echo 5|python act_predict.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/act_predict/; date; echo 10|python act_predict.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/act_predict/; date; echo 15|python act_predict.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/act_predict/; date; echo 20|python act_predict.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/act_predict/; date; echo 25|python act_predict.py; date' | qsub -l nodes=2
echo 'source .bashrc; cd /home/kodama-y/work/act_recog2014/act_predict/; date; echo 30|python act_predict.py; date' | qsub -l nodes=2