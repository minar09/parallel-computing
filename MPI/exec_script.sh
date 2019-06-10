#!/bin/bash

processes=(1 2 4 8)
exec_modes=(0 1)
iterations=(1 2 3 4 5)
matrix_sizes=(100 200 400 600 800 1000)


for exec_mode in "${exec_modes[@]}"
do
    echo "-------------------------------"
    echo "|  Execution mode $exec_mode  |"
    echo "-------------------------------"

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
		    mpirun -n $thread ./jacobi_mpi.o $matrix_size $exec_mode
		    echo "----------------Gauss-Seidel---------------"
		    mpirun -n $thread ./gs_mpi.o $matrix_size $exec_mode

		done
	    done
	done
done

