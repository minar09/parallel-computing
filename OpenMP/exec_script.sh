#!/bin/bash

iterations=(1 2 3 4 5)
processes=(1 2 4 8)
matrix_sizes=(100 200 400 600 800 1000)

for matrix_size in "${matrix_sizes[@]}"
do
    echo "-------------------------------"
    echo "|  Matrix size $matrix_size  |"
    echo "-------------------------------"

    for thread in "${processes[@]}"
    do
        echo "-------------------------------"
        echo "| Processes $thread  |"
        echo "-------------------------------"

	for iteration in "${iterations[@]}"
	do
	    echo "-------------------------------"
	    echo "| Iteration $iteration  |"
	    echo "-------------------------------"
	
	    echo "----------------Jacobi---------------"
	    ./jacobi_openmp.o $matrix_size $thread
	    echo "----------------Gauss-Seidel---------------"
	    ./gs_openmp.o $matrix_size $thread

	done
    done
done
