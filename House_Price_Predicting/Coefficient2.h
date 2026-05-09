#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// Data structure definition (forward declaration)
typedef struct {
    float Size;
    int Location;
    int Security;
    int Garage;
    int Floor_Number;
    int Number_of_Rooms;
    int Built_Year;
    float Price;
} Data;

// Normalization parameters structure
typedef struct {
    float min;
    float max;
} NormalizationParams;

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

// Function to invert a matrix using Gaussian elimination
void matrix_inverse(double **result, double **mat)
{
    int n = 8;
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
        // Find pivot
        double pivot = augmented[i][i];
        if (pivot == 0) {
            // Try to find a non-zero pivot
            for (int k = i + 1; k < n; k++) {
                if (augmented[k][i] != 0) {
                    // Swap rows
                    double *temp_row = augmented[i];
                    augmented[i] = augmented[k];
                    augmented[k] = temp_row;
                    pivot = augmented[i][i];
                    break;
                }
            }
        }
        
        if (pivot == 0) continue; // Skip singular matrix elements
        
        // Make the diagonal contain all 1's
        for (int j = 0; j < 2 * n; j++)
        {
            augmented[i][j] /= pivot;
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
    int m = 8; // 7 features + intercept

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

    matrix_inverse(X_transpose_X_inv, X_transpose_X);

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

// Calculate coefficients from training data (raw data, not normalized)
double *calculateCoefficientsOnTrainingData(Data *trainSet, int trainCount, NormalizationParams *normParams)
{
    int m = 8; // 7 features + intercept
    
    double **X = (double **)malloc(trainCount * sizeof(double *));
    double *y = (double *)malloc(trainCount * sizeof(double));

    // Use raw training data (no normalization for coefficient calculation)
    for (int i = 0; i < trainCount; i++)
    {
        X[i] = (double *)malloc(7 * sizeof(double)); // 7 features
        
        X[i][0] = (double)trainSet[i].Size;
        X[i][1] = (double)trainSet[i].Location;
        X[i][2] = (double)trainSet[i].Security;
        X[i][3] = (double)trainSet[i].Garage;
        X[i][4] = (double)trainSet[i].Floor_Number;
        X[i][5] = (double)trainSet[i].Number_of_Rooms;
        X[i][6] = (double)trainSet[i].Built_Year;
        
        y[i] = (double)trainSet[i].Price;
    }

    double *coefficients = (double *)malloc(sizeof(double) * m);
    calculate_coefficients(X, y, trainCount, coefficients);

    // Free allocated memory
    for (int j = 0; j < trainCount; j++) {
        free(X[j]);
    }
    free(X);
    free(y);

    return coefficients;
}

// Legacy function for backward compatibility (reads all data from file)
double *finalCoefficientCalc()
{
    FILE *file = fopen("house_price_dataset.csv", "r");
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
        X[i] = (double *)malloc(7 * sizeof(double)); // 7 features
        
        // Fixed parsing of CSV
        char *token = strtok(line, ",");
        X[i][0] = atof(token); // Size
        
        token = strtok(NULL, ",");
        X[i][1] = atof(token); // Location
        
        token = strtok(NULL, ",");
        X[i][2] = atof(token); // Security
        
        token = strtok(NULL, ",");
        X[i][3] = atof(token); // Garage
        
        token = strtok(NULL, ",");
        X[i][4] = atof(token); // Floor Number
        
        token = strtok(NULL, ",");
        X[i][5] = atof(token); // Number of Rooms
        
        token = strtok(NULL, ",");
        X[i][6] = atof(token); // Built Year
        
        token = strtok(NULL, ",");
        y[i] = atof(token);    // Price

        i++;
    }

    fclose(file);

    double *coefficients = (double *)malloc(sizeof(double) * 8); // 7 features + intercept
    calculate_coefficients(X, y, n, coefficients);

    // Free allocated memory
    for (int j = 0; j < n; j++) {
        free(X[j]);
    }
    free(X);
    free(y);

    return coefficients;
}