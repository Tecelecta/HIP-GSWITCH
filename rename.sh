#!/bin/sh
rename_loop(){
  for item in `ls`
  do
    if [ -d $item ]
    then
      cd $item
      rename_loop $1 $2
      cd ..
    elif [ ${item#*.} = $1 ]
    then
      mv $item ${item%%.*}.$2
      #sed 's/.cuh/.hxx/g' $item > ${item}.out
    fi
  done
}

rename_loop $1 $2
