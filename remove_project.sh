#!/bin/sh

echo "Removing project $1"

rm -ri "bin/$1";
rm -ri "demo/$1";
rm -ri "data/$1";
rm -ri "networks/$1";
rm -ri "snapshots/$1";
rm -ri "src/projects/$1";

