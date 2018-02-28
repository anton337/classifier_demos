
DIR=`basename ${PWD}`

DATA=$PROJECT_HOME/data/$DIR/slice1.bmp

MASK=$PROJECT_HOME/data/$DIR/mask1.bmp

SNAPSHOTS=$PROJECT_HOME/snapshots/$DIR/

EXECUTABLE=$PROJECT_HOME/bin/$DIR/demo.bin

cat $DATA $MASK $SNAPSHOTS

gdb $EXECUTABLE 

