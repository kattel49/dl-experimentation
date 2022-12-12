#!/bin/bash

files=$(ls /home/aabhushan/.local/share/org.gnome.SoundRecorder)

for f in $files
do
	echo "Moving file: $f"
	mv "/home/aabhushan/.local/share/org.gnome.SoundRecorder/$f" "$f.flac"
done
