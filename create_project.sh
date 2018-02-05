#!/bin/sh

echo "Creating new project $1"

mkdir "bin/$1";
mkdir "demo/$1";
touch "demo/$1/run.sh";
chmod a+x "demo/$1/run.sh";
mkdir "data/$1";
mkdir "networks/$1";
mkdir "snapshots/$1";
mkdir "src/projects/$1";
touch "src/projects/$1/main.cpp";

