
DIR=`basename ${PWD}`

DATA=$PROJECT_HOME/data/$DIR/KO.csv

EXECUTABLE=$PROJECT_HOME/bin/$DIR/demo.bin

$EXECUTABLE $DATA

