#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Matrix multiplication benchmark
// This will be used to test our ML-based optimization

#define SIZE 1000

void matrix_multiply(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void initialize_matrix(double M[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            M[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

int main() {
    // Allocate matrices
    double (*A)[SIZE] = malloc(SIZE * SIZE * sizeof(double));
    double (*B)[SIZE] = malloc(SIZE * SIZE * sizeof(double));
    double (*C)[SIZE] = malloc(SIZE * SIZE * sizeof(double));
    
    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize with random values
    srand(time(NULL));
    initialize_matrix(A);
    initialize_matrix(B);
    
    // Measure time
    clock_t start = clock();
    
    // Perform matrix multiplication
    matrix_multiply(A, B, C);
    
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    // Print execution time
    printf("Matrix multiplication time: %f seconds\n", cpu_time_used);
    
    // Print a sample result to prevent optimization from removing the computation
    printf("Sample result C[0][0] = %f\n", C[0][0]);
    
    // Free memory
    free(A);
    free(B);
    free(C);
    
    return 0;
}
