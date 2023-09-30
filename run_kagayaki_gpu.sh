#!/bin/tcsh

######################################################
# Running batch job in JAIST kagayaki server         #
# Example:                                           #
#   If in local computer:                            #
#    $  conda activate env                           #
#    $  ./run.sh script.py --arg1 10 --arg2 20       #
#   Then this script will be used as                 #
#    $  qsub -q <QUEUE_NAME> -v CONDA_ENV_NAME \     #
#      "env",ARGS="script.py --arg1 10 --arg2 20", \ #
#      LOGFILE="logs.txt" run_kagayaki_gpu.sh        #
######################################################


#PBS -N se-vqvae
#PBS -j oe

# Change dir to the repo dir
cd ${PBS_O_WORKDIR}

# Get queue name
set QUEUE_NAME = "${PBS_QUEUE}"
echo "Queue: $QUEUE_NAME" >> $LOGFILE

# Load necessary modules
source /etc/profile.d/modules.csh
module purge

# Load CUDA if the queue is GPU
set gpu_cond_check = `expr $QUEUE_NAME : '^GPU-.*$'`
set n = `echo $QUEUE_NAME | wc -c`
@ n--
if ($gpu_cond_check == $n) then
  module load cuda
  echo "CUDA loaded!" >> $LOGFILE
else
  echo "Run on CPU!" >> $LOGFILE
endif

# Allow local conda
source $HOME/conda/x64/etc/profile.d/conda.csh

# Activate environment
conda activate $CONDA_ENV_NAME
echo "Conda prefix: $CONDA_PREFIX" >> $LOGFILE

# Add repo directory to local path
# setenv PYTHONPATH "`pwd`"
# echo "Set PYTHONPATH=$PYTHONPATH" >> $LOGFILE

# echo "Command: python $ARGS" >> $LOGFILE

# Split ARGS by space (?)
set SPLIT_ARGS = ($ARGS:as/,/ /)
./run.sh $SPLIT_ARGS
# python $SPLIT_ARGS