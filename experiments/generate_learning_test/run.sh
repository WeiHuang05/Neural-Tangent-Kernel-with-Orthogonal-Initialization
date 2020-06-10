#!/bin/bash



for i in {0..19}
do
	for lr in {0..28}


	do
		python train.py $i $lr
	
	done
done


   
