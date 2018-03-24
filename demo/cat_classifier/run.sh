
DIR=`basename ${PWD}`

DATA=$PROJECT_HOME/data/$DIR/kittie-crazy-desktop-black-and-white.bmp

SNAPSHOTS=$PROJECT_HOME/snapshots/$DIR/

EXECUTABLE=$PROJECT_HOME/bin/$DIR/demo.bin

$EXECUTABLE $DATA $SNAPSHOTS

