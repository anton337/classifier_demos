
DIR=`basename ${PWD}`

DATA=$PROJECT_HOME/data/$DIR/angola.bmp

EXECUTABLE=$PROJECT_HOME/bin/$DIR/demo.bin

echo $DATA

gdb $EXECUTABLE

