
DIR=`basename ${PWD}`

DATA_IMAGE=$PROJECT_HOME/data/$DIR/emnist-letters-train-images-idx3-ubyte

DATA_LABEL=$PROJECT_HOME/data/$DIR/emnist-letters-train-labels-idx1-ubyte

SNAPSHOTS=$PROJECT_HOME/snapshots/$DIR/

EXECUTABLE=$PROJECT_HOME/bin/$DIR/demo.bin

od -i --endian=big -N 600 $DATA_IMAGE

$EXECUTABLE $DATA_IMAGE $DATA_LABEL $SNAPSHOTS

