# !/bin/bash

# This script converts videos into frames
# for different fps change (-r 1)
# put this file under video folder


for f in *.avi
  do g=`echo $f | sed 's/\.avi//'`;
  echo Processing $f; 
  mkdir -p frames/$g/ ;
  ffmpeg -i $f frames/$g/%06d.jpeg ; 
done
