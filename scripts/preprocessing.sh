export DATA_DIR=$HOME/DATA
export SCRIPTS_DIR=$HOME/LiverSeg/scripts
export SRC_DATA_DIR=$DATA_DIR/original_data
export PREPROCESSED_DATA_DIR=$DATA_DIR/preprocessed_data
export TRAIN_DIR=$DATA_DIR/training_data

#$PYTHON $SCRIPTS_DIR/preprocess.py -i $SRC_DATA_DIR -t $PREPROCESSED_DATA_DIR -f
$PYTHON $SCRIPTS_DIR/create_train_test_dir.py -i $PREPROCESSED_DATA_DIR -t $TRAIN_DIR -f