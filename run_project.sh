#!/bin/sh

PROJECT_HOME=`pwd`

echo "Project home: "$PROJECT_HOME

export PROJECT_HOME=$PROJECT_HOME

echo "Running project $1"

cd $PROJECT_HOME/demo/$1/

./run.sh

