#include<stdio.h>

int main( int ac, char **av)

{

	#pragma omp parallel // specify the code between the curly brackets is part of an OpenMP 	parallel section.

	{

		printf("Hello World!!!\n");

	}

	return 0;

}
