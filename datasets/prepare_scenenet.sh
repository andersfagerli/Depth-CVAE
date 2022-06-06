#!/bin/bash

cd SceneNetRGBD/train_0/train

mkdir depth/ instance/ photo/

WD=$(pwd)
COUNTER=0

for d in 16/*
do
  cd $WD/$d/depth

  TEMP=$COUNTER
  for file in *.png
  do
    #echo "Renaming $file to $COUNTER.png ..."
    mv "$file" "temp$COUNTER.png"
    #echo "Moving $COUNTER.png to folder ..."
    mv "temp$COUNTER.png" ../../../depth 
    COUNTER=$[$COUNTER+1]
  done
  cd ../../../
  
  cd $WD/$d/photo
  COUNTER=$TEMP
  for file in *.jpg
  do
    mv "$file" "$COUNTER.png"
    mv $COUNTER.png ../../../photo
    COUNTER=$[$COUNTER+1]
  done
  cd ../../../

  echo "Images moved: $COUNTER"
done

rm -rf 0/

echo "Renaming images ..."

cd depth

for file in *.png; do mv "$file" "${file/temp/}"; done

cd ../../../../
