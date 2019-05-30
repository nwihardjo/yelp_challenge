#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -N train_model
#PBS -e ./train_model.txt
#PBS -o ./train_model.txt
cd "$HOME/Project_1"
echo Starting Programme
python best_5features.py
echo Ending Programme
