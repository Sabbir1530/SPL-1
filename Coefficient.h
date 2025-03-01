#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Function to transpose a matrix
void matrix_transpose(double **result, double **mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[j][i] = mat[i][j];
        }
    }
}

// Function to multiply two matrices
void matrix_multiply(double **result, double **mat1, double **mat2, int rows1, int cols1, int cols2)
{
    for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < cols2; j++)
        {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++)
            {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

// Function to invert a 4x4 matrix using Gaussian elimination
void matrix_inverse_4x4(double **result, double **mat)
{
    int n = 4;
    double temp;

    // Create an augmented matrix with the identity matrix
    double **augmented = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
    {
        augmented[i] = (double *)malloc(2 * n * sizeof(double));
        for (int j = 0; j < n; j++)
        {
            augmented[i][j] = mat[i][j];
            augmented[i][j + n] = (i == j) ? 1 : 0;
        }
    }

    // Perform Gaussian elimination
    for (int i = 0; i < n; i++)
    {
        // Make the diagonal contain all 1's
        temp = augmented[i][i];
        for (int j = 0; j < 2 * n; j++)
        {
            augmented[i][j] /= temp;
        }

        // Make the other rows contain 0's
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                temp = augmented[j][i];
                for (int k = 0; k < 2 * n; k++)
                {
                    augmented[j][k] -= augmented[i][k] * temp;
                }
            }
        }
    }

    // Extract the inverse matrix
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i][j] = augmented[i][j + n];
        }
    }

    for (int i = 0; i < n; i++)
    {
        free(augmented[i]);
    }
    free(augmented);
}

// Function to calculate the coefficients
void calculate_coefficients(double **X, double *y, int n, double *coefficients)
{
    int m = 5;

    double **X_augmented = (double **)malloc(n * sizeof(double *));
    double **X_transpose = (double **)malloc(m * sizeof(double *));
    double **X_transpose_X = (double **)malloc(m * sizeof(double *));
    double **X_transpose_X_inv = (double **)malloc(m * sizeof(double *));
    double **X_transpose_X_inv_X_transpose = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < n; i++)
    {
        X_augmented[i] = (double *)malloc(m * sizeof(double));
    }
    for (int i = 0; i < m; i++)
    {
        X_transpose[i] = (double *)malloc(n * sizeof(double));
        X_transpose_X[i] = (double *)malloc(m * sizeof(double));
        X_transpose_X_inv[i] = (double *)malloc(m * sizeof(double));
        X_transpose_X_inv_X_transpose[i] = (double *)malloc(n * sizeof(double));
    }

    for (int i = 0; i < n; i++)
    {
        X_augmented[i][0] = 1.0;
        for (int j = 1; j < m; j++)
        {
            X_augmented[i][j] = X[i][j - 1];
        }
    }

    matrix_transpose(X_transpose, X_augmented, n, m);

    matrix_multiply(X_transpose_X, X_transpose, X_augmented, m, n, m);

    matrix_inverse_4x4(X_transpose_X_inv, X_transpose_X);

    matrix_multiply(X_transpose_X_inv_X_transpose, X_transpose_X_inv, X_transpose, m, m, n);

    for (int i = 0; i < m; i++)
    {
        coefficients[i] = 0;
        for (int j = 0; j < n; j++)
        {
            coefficients[i] += X_transpose_X_inv_X_transpose[i][j] * y[j];
        }
    }

    for (int i = 0; i < n; i++)
    {
        free(X_augmented[i]);
    }
    for (int i = 0; i < m; i++)
    {
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

double *finalCoefficientCalc()
{
    FILE *file = fopen("Processed_Dataset.csv", "r");
    if (!file)
    {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    char line[256];
    int n = 0;
    while (fgets(line, sizeof(line), file))
    {
        n++;
    }
    n--; // Subtract header row

    fseek(file, 0, SEEK_SET);
    fgets(line, sizeof(line), file); // Skip header

    double **X = (double **)malloc(n * sizeof(double *));
    double *y = (double *)malloc(n * sizeof(double));

    int i = 0;
    while (fgets(line, sizeof(line), file))
    {
        X[i] = (double *)malloc(4 * sizeof(double));
        char *token = strtok(line, ",");
        token = strtok(NULL, ",");         // Skip Date
        token = strtok(NULL, ",");         // Skip Season
        y[i] = atof(strtok(NULL, ","));    // Temperature
        X[i][0] = atof(strtok(NULL, ",")); // Humidity
        X[i][1] = atof(strtok(NULL, ",")); // Wind_Speed
        X[i][2] = atof(strtok(NULL, ",")); // Pressure
        X[i][3] = atof(strtok(NULL, ",")); // Rain
        i++;
    }

    fclose(file);

    double *coefficients;
    coefficients = (double *)malloc(sizeof(double) * 5);
    calculate_coefficients(X, y, n, coefficients);


    double *data = (double*)malloc(sizeof(double)*3);
    data[0] = coefficients[1];
    data[1] = coefficients[2];
    data[2] = coefficients[3]/1000;

    // for(int i=0;i<5;i++) printf("%lf ", coefficients[i]);
    // printf("\n");
    return data;
}
