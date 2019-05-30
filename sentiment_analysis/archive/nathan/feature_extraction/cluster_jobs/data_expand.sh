#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -N data_expand
#PBS -e ./data_expand_model.txt
#PBS -o ./data_expand_model.txt
cd "$HOME/Project_1"
echo Starting Programme
python expand.py
echo Ending Programme
