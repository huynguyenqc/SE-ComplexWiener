#!/bin/sh
export PYTHONPATH=$(pwd)
echo "PYTHONPATH=$PYTHONPATH"
python "$@"