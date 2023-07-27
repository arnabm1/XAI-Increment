#!/bin/tcsh

set VENV="audio_ml_venv"
set PROJ_NAME="audio_ml"
set CUDA="10.1"
set GPU="num=1:j_exclusive=yes:gmodel=TeslaP4"
set PY_PATH="/home/$PROJ_NAME/virtualenvs/$VENV/bin/python"

set FILE_PATH="/ifxhome/mazumderarna/XAI/XAI_Gitlab/infexplain/research/EUSIPCO_Paper/modular_codes/IL_train.py"

module load lsf
module load cuda/$CUDA

#ewc logs path
set EWC_LOG_PATH_1 = "/home/audio_ml.work/data/audio/speech/speech_commands_arrays/Trained_Models/EUSIPCO/ewc/Logs/run_1.txt"
set EWC_LOG_PATH_2 = "/home/audio_ml.work/data/audio/speech/speech_commands_arrays/Trained_Models/EUSIPCO/ewc/Logs/run_2.txt"
set EWC_LOG_PATH_3 = "/home/audio_ml.work/data/audio/speech/speech_commands_arrays/Trained_Models/EUSIPCO/ewc/Logs/run_3.txt"

set A="--sessions=16"
set B="--mode=ewc"
set C1="--folder=run_1"
set C2="--folder=run_2"
set C3="--folder=run_3"
set D1="--seed=10"
set D2="--seed=20"
set D3="--seed=30"

set ARGS="$A $B $C1 $D1"
bsub -Is -q batch -R "osrel==70 && ui==aiml_batch_training" -n 8 -P $PROJ_NAME $PY_PATH $FILE_PATH $ARGS > $EWC_LOG_PATH_1 &
set ARGS="$A $B $C2 $D2"
bsub -Is -q batch -R "osrel==70 && ui==aiml_batch_training" -n 8 -P $PROJ_NAME $PY_PATH $FILE_PATH $ARGS > $EWC_LOG_PATH_2 &
set ARGS="$A $B $C3 $D3"
bsub -Is -q batch -R "osrel==70 && ui==aiml_batch_training" -n 8 -P $PROJ_NAME $PY_PATH $FILE_PATH $ARGS > $EWC_LOG_PATH_3

