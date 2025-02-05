#include <stdio.h>
#include <stdlib.h>

void matrix_multiply(double **result, double **mat1, double **mat2, int rows1, int cols1, int cols2) {
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

void matrix_transpose(double **result, double **mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = mat[i][j];
        }
    }
}

void matrix_inverse_4x4(double **result, double **mat) {
       for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i][j] = (i == j) ? 1 : 0; 
        }
    }
}

void calculate_coefficients(double **X, double *y, int n, double *coefficients) {
    int m = 4; 

    double **X_augmented = (double **)malloc(n * sizeof(double *));
    double **X_transpose = (double **)malloc(m * sizeof(double *));
    double **X_transpose_X = (double **)malloc(m * sizeof(double *));
    double **X_transpose_X_inv = (double **)malloc(m * sizeof(double *));
    double **X_transpose_X_inv_X_transpose = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < n; i++) {
        X_augmented[i] = (double *)malloc(m * sizeof(double));
    }
    for (int i = 0; i < m; i++) {
        X_transpose[i] = (double *)malloc(n * sizeof(double));
        X_transpose_X[i] = (double *)malloc(m * sizeof(double));
        X_transpose_X_inv[i] = (double *)malloc(m * sizeof(double));
        X_transpose_X_inv_X_transpose[i] = (double *)malloc(n * sizeof(double));
    }

    for (int i = 0; i < n; i++) {
        X_augmented[i][0] = 1.0;
        for (int j = 1; j < m; j++) {
            X_augmented[i][j] = X[i][j - 1];
        }
    }

    matrix_transpose(X_transpose, X_augmented, n, m);

    matrix_multiply(X_transpose_X, X_transpose, X_augmented, m, n, m);

    matrix_inverse_4x4(X_transpose_X_inv, X_transpose_X);

    matrix_multiply(X_transpose_X_inv_X_transpose, X_transpose_X_inv, X_transpose, m, m, n);

    for (int i = 0; i < m; i++) {
        coefficients[i] = 0;
        for (int j = 0; j < n; j++) {
            coefficients[i] += X_transpose_X_inv_X_transpose[i][j] * y[j];
        }
    }

    for (int i = 0; i < n; i++) {
        free(X_augmented[i]);
    }
    for (int i = 0; i < m; i++) {
        free(X_transpose[i]);
        free(X_transpose_X[i]);
        free(X_transpose_X_inv[i]);
        free(X_transpose_X_inv_X_transpose[i]);
    }
    free(X_augmented);
    free(X_transpose);
    free(X_transpose_X);
    free(X_transpose_X_inv);
    free(X_transpose_X_inv_X_transpose);
}

int main() {
    int n = 5; // Number of samples
    double *X[] = {
        (double[]){1.0, 2.0, 3.0},
        (double[]){2.0, 3.0, 4.0},
        (double[]){3.0, 4.0, 5.0},
        (double[]){4.0, 5.0, 6.0},
        (double[]){5.0, 6.0, 7.0}
    };
    double y[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double coefficients[4];

    calculate_coefficients(X, y, n, coefficients);

    printf("Coefficients:\n");
    for (int i = 0; i < 4; i++) {
        printf("%f\n", coefficients[i]);
    }

    return 0;
}