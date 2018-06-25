#!/bin/bash

PROJECT_PATH=`pwd`
DATA_PATH="${PROJECT_PATH}/data"
SRC_PATH="${PROJECT_PATH}/src"
MODEL_PATH="${PROJECT_PATH}/models"
LOG_PATH="${PROJECT_PATH}/logs"

REQUIREMENTS_PATH="${SRC_PATH}/requirements.txt"
PARAMS_PATH="${SRC_PATH}/params.yaml"
BLIND_SET=""	# for real time test data 
pip_version=`which pip`
python_version=`which python`
tensorboard_cmd=`which tensorboard`

start_block () {
	echo $1 starts ...
}

close_block () {
	echo $1 ends ...
}


BLOCK_TITLE="Model Training"
start_block "${BLOCK_TITLE}"
pip_version install -r ${REQUIREMENTS_PATH};
tensorboard_cmd ${LOG_PATH} &;
python_version "${SRC_PATH}/main.py" \
--data_dir ${DATA_PATH} --model_dir ${MODEL_PATH} --log_dir ${LOG_PATH} --params_path ${PARAMS_PATH}
close_block "${BLOCK_TITLE}"