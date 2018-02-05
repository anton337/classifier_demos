#!/bin/sh

PROJECT_HOME=`pwd`

echo "Project home: "$PROJECT_HOME

export PROJECT_HOME=$PROJECT_HOME

echo "Project: "$PROJECT

cd src;
make $1;

