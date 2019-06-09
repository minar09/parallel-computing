#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"

#define MAX_ITER 100
#define MAX 100 //maximum value of the matrix element
#define TOL 0.000001

// Generate a random float number with the maximum value of max
float rand_float(int max){
  return ((float)rand()/(float)(RAND_MAX)) * max;
}


// Allocate 2D matrix
void allocate_init_2Dmatrix(float ***mat, int n, int m){
  int i, j;
  *mat = (float **) malloc(n * sizeof(float *));
  for(i = 0; i < n; i++) {
    (*mat)[i] = (float *)malloc(m * sizeof(float));
    for (j = 0; j < m; j++)
      (*mat)[i][j] = rand_float(MAX);
  }
} 

// solver
void solver(float ***mat, int n, int m){
  float diff = 0, temp;
  int done = 0, cnt_iter = 0, i, j;

  while (!done && (cnt_iter < MAX_ITER)){
    diff = 0;
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++){
	temp = (*mat)[i][j];
	(*mat)[i][j] = 0.2 * ((*mat)[i][j] + (*mat)[i][j - 1] + (*mat)[i - 1][j] + (*mat)[i][j + 1] + (*mat)[i + 1][j]);
	diff += abs((*mat)[i][j] - temp);
      }
    if (diff/n/n < TOL)
      done = 1; 
    cnt_iter ++;
  }

  if (done)
    printf("Solver converged after %d iterations\n", cnt_iter);
  else
    printf("Solver not converged after %d iterations\n", cnt_iter);
  
}

int main(int argc, char *argv[]) {
  int n, communication;
  float **a;
  
  MPI_Init(&argc, &argv);

  if (argc < 3) {
    printf("Call this program with two parameters: matrix_size communication \n");
    printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");
    printf("\t communication:\n");
    printf("\t\t 0: initial and final using point-to-point communication\n");
    printf("\t\t 1: initial and final using collective communication\n");
    
    exit(1);
  }

  n = atoi(argv[1]);
  communication =  atoi(argv[2]);
  printf("Matrix size = %d communication = %d\n", n, communication);

  allocate_init_2Dmatrix(&a, n, n);

  double tsop = MPI_Wtime();
  solver(&a, n, n);
  double tfop = MPI_Wtime();

  FILE *f;
		if (access("seq.csv", F_OK) == -1) {
 			f = fopen("seq.csv", "a");
			fprintf(f, "Operations-time\n");
		}
		else {
			f = fopen("seq.csv", "a");
		}

		fprintf(f, "%f;\n", tfop - tsop);
		fclose(f);

  MPI_Finalize();
  return 0;
}
