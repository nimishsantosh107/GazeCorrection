#!/bin/sh

## TO RUN FROM PROJECT ROOT ##

mkdir GAZE_BUNDLE

cp -r ./dataset ./GAZE_BUNDLE &&
cp -r ./TEST_INPUT ./GAZE_BUNDLE &&
cp -r ./TRAINING ./GAZE_BUNDLE &&

zip -r GAZE.zip GAZE_BUNDLE &&
rm -rf GAZE_BUNDLE

echo "DONE"
