#!/bin/sh

PROJECT_HOME=`pwd`

echo "Project home: "$PROJECT_HOME

export PROJECT_HOME=$PROJECT_HOME

export PROJECT=simple_ann_classification

echo "Running project $PROJECT"

cd $PROJECT_HOME/demo/$PROJECT/

./run.sh

