#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"

#define MAX_ITER 10000

// Maximum value of the matrix element
#define MAX 10000
#define TOL 0.000001




// Generate a random float number with the maximum value of max
float rand_float(int max) {
	return ((float)rand() / (float)(RAND_MAX)) * max;
}




// Calculates how many rows are given, as maximum, to each node
int get_max_rows(int num_nodes, int n) {
	return (int)(ceil((n-2) / num_nodes) + 2);
}




// Gets the position from which elements are gonna be sent / received
int get_node_offset(int node_id, int n, int max_rows) {
	return node_id * n * (max_rows-2);
}




// Calculates how many elements are going to a given node
int get_node_elems(int node_id, int n, int max_rows) {

	int node_offset = get_node_offset(node_id, n, max_rows);
	int node_elems = max_rows * n;

	// Case in which the node receive the full set of elements
	if (node_offset + node_elems <= (n*n)) {
		return node_elems;
	}

	// Case of the last node, which could get less elements
	else {
		return (n*n) - node_offset;
	}
}




// Allocate 2D matrix in the master node
void allocate_root_matrix(float **mat, int n, int m){

	*mat = (float *) malloc(n * m * sizeof(float));
	for (int i = 0; i < (n*m); i++) {
		(*mat)[i] = rand_float(MAX);
	}
}




// Allocate 2D matrix in the slaves nodes
void allocate_node_matrix(float **mat, int num_elems) {
	*mat = (float *) malloc(num_elems * sizeof(float));
}




// Solves as many elements as specified in "num_elems"
int solver(float **mat, int n, int num_elems) {

	float diff = 0, temp;
	int done = 0, cnt_iter = 0, myrank;

  	while (!done && (cnt_iter < MAX_ITER)) {
  		diff = 0;
		
		// define temp array
		float temp[n*n];

  		// Neither the first row nor the last row are solved
  		// (that's why it starts at "n" and it goes up to "num_elems - 2n")
  		for (int i = n; i < num_elems - (2*n); i++) {

  			// Additionally, neither the first nor last column are solved
  			// (that's why the first and last positions of "rows" are skipped)
  			if ((i % n == 0) || (i+1 % n == 0)) {
				continue;
			}

  			int pos_up = i - n;
  			int pos_do = i + n;
  			int pos_le = i - 1;
  			int pos_ri = i + 1;

			temp[i] = 0.25 * ((*mat)[i] + (*mat)[pos_le] + (*mat)[pos_up] + (*mat)[pos_ri] + (*mat)[pos_do]);
			diff += abs((*mat)[i] - temp[i]);
      	}

		if (diff/n/n < TOL) {
			done = 1;
		}
		cnt_iter ++;

		// substitue new values
		for (int i = n; i < num_elems - (2*n); i++) {

			(*mat)[i] = temp[i];

		}
	}


	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if (done) {
		printf("Node %d: Solver converged after %d iterations\n", myrank, cnt_iter);
	}
	else {
		printf("Node %d: Solver not converged after %d iterations\n", myrank, cnt_iter);
	}

	return cnt_iter;
}




int main(int argc, char *argv[]) {

	int np, myrank, n, communication, i;
	float *a, *b;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	if (argc < 3) {
		if (myrank == 0) {
			printf("Call this program with two parameters: matrix_size communication \n");
			printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");
			printf("\t communication:\n");
			printf("\t\t 0: initial and final using point-to-point communication\n");
			printf("\t\t 1: initial and final using collective communication\n");
		}

		MPI_Finalize();
		exit(1);
	}


	n = atoi(argv[1]);
	communication =	atoi(argv[2]);
	
	if (myrank == 0) {
		printf("Matrix size = %d communication = %d\n", n, communication);
	}


	// Calculate common relevant values for each node
	int max_rows = get_max_rows(np, n);

	// Array of containing the offset and number of elements per node
	int nodes_offsets[np];
	int nodes_elems[np];
	for (i = 0; i < np; i++) {
		nodes_offsets[i] = get_node_offset(i, n, max_rows);
		nodes_elems[i] = get_node_elems(i, n, max_rows);
	}

	// Variable to store the node local number of elements
	int num_elems = nodes_elems[myrank];


	double tscom1 = MPI_Wtime();


	switch(communication) {
		case 0: {
			if (myrank == 0) {

				// Allocating memory for the whole matrix
				allocate_root_matrix(&a, n, n);

				// Master sends chuncks to every other node
				for (i = 1; i < np; i++) {
					int i_offset = nodes_offsets[i];
					int i_elems = nodes_elems[i];
					MPI_Send(&a[i_offset], i_elems, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
				}
			}
			else {

				// Allocating the exact memory to the rows receiving
				allocate_node_matrix(&a, num_elems);
				MPI_Status status;

				// Receiving the data from the master node
				MPI_Recv(a, num_elems, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			}
			break;
		}

		case 1: {
			if (myrank == 0) {
				// Allocating memory for the whole matrix
				allocate_root_matrix(&a, n, n);		
			}
			// Allocating the exact memory where the receiving rows are computed
			allocate_node_matrix(&b, num_elems);

			// Collective communication for scattering the matrix
			// Info: https://www.mpich.org/static/docs/v3.1/www3/MPI_Scatterv.html
			MPI_Scatterv(a, nodes_elems, nodes_offsets, MPI_FLOAT, b, num_elems, MPI_FLOAT, 0, MPI_COMM_WORLD);
			break;
		}
	}


	double tfcom1 = MPI_Wtime();
	double tsop = MPI_Wtime();
	int num_iter = 0;

	// --------- SOLVER ---------
	if (communication == 0) {
		num_iter = solver(&a, n, num_elems);
	}
	else {
		num_iter = solver(&b, n, num_elems);
	}

	double tfop = MPI_Wtime();
	double tscom2 = MPI_Wtime();


	switch(communication) {
		case 0: {
			if (myrank == 0) {

				MPI_Status status;

				// Master sends chuncks to every other node
				for (i = 1; i < np; i++) {
					int i_offset = nodes_offsets[i] + n; 		// +n to skip cortex values
					int i_elems = nodes_elems[i] - (2*n);		// -2n to skip cortex values
					MPI_Recv(&a[i_offset], i_elems, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				}
			}
			else {
				int solved_offset = n;							// Start at n to skip cortex values
				int solved_elems = num_elems - (2*n);			// Reach num_elems-2n to skip cortex values

				// Compute num_elems sin la corteza
				MPI_Send(&a[solved_offset], solved_elems, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}

			break;
		}
		
		case 1: {

			// Collective communication for gathering the matrix
			// Info: http://www.mpich.org/static/docs/v3.2.1/www/www3/MPI_Gatherv.html
			MPI_Gatherv(b, num_elems, MPI_FLOAT, a, nodes_elems, nodes_offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
			break;
		}
	}

	double tfcom2 = MPI_Wtime();


	if (myrank == 0) {
		float com_time = (tfcom1-tscom1) + (tfcom2-tscom2);
		float ops_time = tfop - tsop;
		float total_time = com_time + ops_time;

		printf("Communication time: %f\n", com_time);
		printf("Operations time: %f\n", ops_time);
		printf("Total time: %f\n", total_time);

		FILE *f;
		if (access("jacobi_mpi_results.csv", F_OK) == -1) {
 			f = fopen("jacobi_mpi_results.csv", "a");
			fprintf(f, "Communication;Nodes;No. of iterations;Size;Communication-time;Operations-time;Total-time;\n");
		}
		else {
			f = fopen("jacobi_mpi_results.csv", "a");
		}

		fprintf(f, "%d;%d;%d;%d;%f;%f;%f;\n", communication, np, num_iter, n, com_time, ops_time, total_time);
		fclose(f);
	}


	free(a);
	free(b);

	MPI_Finalize();
	return 0;
}
