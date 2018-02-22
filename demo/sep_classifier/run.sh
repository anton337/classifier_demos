
DIR=`basename ${PWD}`

DATA_IMAGE=/home/antonk/data/oxy.hdr

DATA_LABEL=/home/antonk/data/afi.hdr

SNAPSHOTS=$PROJECT_HOME/snapshots/$DIR/

EXECUTABLE=$PROJECT_HOME/bin/$DIR/demo.bin

$EXECUTABLE $DATA_IMAGE $DATA_LABEL $SNAPSHOTS

