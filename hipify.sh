#!/bin/sh
hipify=~/HIP-master/hipify-clang/build/hipify-clang
root=~/GSWITCH
target_suffix=

proc_dir() 
{
if [ ! -x $1 ]
then
	exit 1
fi

echo "in dir $1:"
for file in `ls $1`
do
	echo -n "$1/$file is a: "
	if [ -d $1/$file ]
	then
		echo "dir"
		mkdir $file && cd $file
		proc_dir $1/$file
		cd ..
	elif [ ${file##*.} = $target_suffix ]
	then
		echo -n "target "
		$hipify -o-dir=./ $1/$file -I$root/src -I$root/deps/moderngpu/src 2> $file.log && echo --- success || echo --- failed
	else
		echo "normal file"
	fi
done
}

target_suffix=$2
mkdir $1 && cd $1
proc_dir $root/$1
