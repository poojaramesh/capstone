#!/usr/bin/env zsh 

search_path="/home/ubuntu/tar_files"
for f in "search_path"/*.tar.gz;
do
	echo "$f"
	tar -xkf "$f" &> log
done

