
DIR=`basename ${PWD}`

DATA=$PROJECT_HOME/data/$DIR/numbers.bmp
#DATA=$PROJECT_HOME/data/$DIR/digits_mnist.bmp

SNAPSHOTS=$PROJECT_HOME/snapshots/$DIR/

EXECUTABLE=$PROJECT_HOME/bin/$DIR/demo.bin

$EXECUTABLE $DATA $SNAPSHOTS

