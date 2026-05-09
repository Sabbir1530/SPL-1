#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// Structure to store feature normalization statistics
typedef struct {
    double hum_mean, hum_std;
    double wind_mean, wind_std;
    double pres_mean, pres_std;
} NormStats;

// Global normalization statistics
extern NormStats temp_norm_stats;

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
    FILE *file = fopen("Processed_Datase.csv", "r");
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
        X[i] = (double *)malloc(3 * sizeof(double));  // Only 3 features: Humidity, Wind_Speed, Pressure
        strtok(line, ",");                  // Skip Date
        strtok(NULL, ",");                  // Skip Season
        y[i] = atof(strtok(NULL, ","));     // Temperature (output for regression)
        X[i][0] = atof(strtok(NULL, ",")); // Humidity
        X[i][1] = atof(strtok(NULL, ",")); // Wind_Speed
        X[i][2] = atof(strtok(NULL, ",")); // Pressure (keep original scale)
        strtok(NULL, ",");                  // Skip Rain (not used for temperature prediction)
        i++;
    }

    fclose(file);

    // Calculate feature statistics for normalization
    double hum_mean = 0, wind_mean = 0, pres_mean = 0;
    double hum_std = 0, wind_std = 0, pres_std = 0;
    
    for (int i = 0; i < n; i++)
    {
        hum_mean += X[i][0];
        wind_mean += X[i][1];
        pres_mean += X[i][2];
    }
    hum_mean /= n;
    wind_mean /= n;
    pres_mean /= n;
    
    // Calculate standard deviations
    for (int i = 0; i < n; i++)
    {
        hum_std += (X[i][0] - hum_mean) * (X[i][0] - hum_mean);
        wind_std += (X[i][1] - wind_mean) * (X[i][1] - wind_mean);
        pres_std += (X[i][2] - pres_mean) * (X[i][2] - pres_mean);
    }
    hum_std = sqrt(hum_std / n);
    wind_std = sqrt(wind_std / n);
    pres_std = sqrt(pres_std / n);
    
    if (hum_std < 0.1) hum_std = 1.0;
    if (wind_std < 0.1) wind_std = 1.0;
    if (pres_std < 0.1) pres_std = 1.0;

    // Normalize features for better regression
    for (int i = 0; i < n; i++)
    {
        X[i][0] = (X[i][0] - hum_mean) / hum_std;    // Normalize humidity
        X[i][1] = (X[i][1] - wind_mean) / wind_std;  // Normalize wind speed
        X[i][2] = (X[i][2] - pres_mean) / pres_std;  // Normalize pressure
    }

    double *coefficients = (double *)malloc(sizeof(double) * 4);  // 4 coefficients: intercept + 3 features
    calculate_coefficients(X, y, n, coefficients);

    // Store normalization statistics in global struct for use during prediction
    temp_norm_stats.hum_mean = hum_mean;
    temp_norm_stats.hum_std = hum_std;
    temp_norm_stats.wind_mean = wind_mean;
    temp_norm_stats.wind_std = wind_std;
    temp_norm_stats.pres_mean = pres_mean;
    temp_norm_stats.pres_std = pres_std;

    // Free temporary arrays
    for (int i = 0; i < n; i++)
    {
        free(X[i]);
    }
    free(X);
    free(y);

    return coefficients;  // Return all 4 coefficients directly
}

// Train a separate logistic regression model for rain prediction
double *trainRainModel()
{
    FILE *file = fopen("Processed_Datase.csv", "r");
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
        X[i] = (double *)malloc(3 * sizeof(double));
        strtok(line, ",");                  // Skip Date
        strtok(NULL, ",");                  // Skip Season
        strtok(NULL, ",");                  // Skip Temperature
        X[i][0] = atof(strtok(NULL, ",")); // Humidity
        X[i][1] = atof(strtok(NULL, ",")); // Wind_Speed
        X[i][2] = atof(strtok(NULL, ",")); // Pressure
        y[i] = atof(strtok(NULL, ","));    // Rain (output for logistic regression)
        i++;
    }

    fclose(file);

    // Calculate means and standard deviations for feature normalization
    double hum_mean = 0, wind_mean = 0, pres_mean = 0;
    double hum_std = 0, wind_std = 0, pres_std = 0;
    
    for (int i = 0; i < n; i++)
    {
        hum_mean += X[i][0];
        wind_mean += X[i][1];
        pres_mean += X[i][2];
    }
    hum_mean /= n;
    wind_mean /= n;
    pres_mean /= n;
    
    for (int i = 0; i < n; i++)
    {
        hum_std += (X[i][0] - hum_mean) * (X[i][0] - hum_mean);
        wind_std += (X[i][1] - wind_mean) * (X[i][1] - wind_mean);
        pres_std += (X[i][2] - pres_mean) * (X[i][2] - pres_mean);
    }
    hum_std = sqrt(hum_std / n);
    wind_std = sqrt(wind_std / n);
    pres_std = sqrt(pres_std / n);
    
    if (hum_std < 0.01) hum_std = 1.0;
    if (wind_std < 0.01) wind_std = 1.0;
    if (pres_std < 0.01) pres_std = 1.0;

    // Calculate rain probability statistics
    double rain_mean = 0;
    for (int i = 0; i < n; i++)
    {
        rain_mean += y[i];
    }
    rain_mean /= n;
    
    // Calculate correlation coefficients between features and rain outcome
    double cov_hum = 0, cov_wind = 0, cov_pres = 0;
    for (int i = 0; i < n; i++)
    {
        cov_hum += ((X[i][0] - hum_mean) / hum_std) * (y[i] - rain_mean);
        cov_wind += ((X[i][1] - wind_mean) / wind_std) * (y[i] - rain_mean);
        cov_pres += ((X[i][2] - pres_mean) / pres_std) * (y[i] - rain_mean);
    }
    cov_hum /= n;
    cov_wind /= n;
    cov_pres /= n;
    
    // Convert probability to log-odds for intercept
    double rain_prob = rain_mean;
    if (rain_prob < 0.01) rain_prob = 0.01;
    if (rain_prob > 0.99) rain_prob = 0.99;
    double intercept = log(rain_prob / (1 - rain_prob));
    
    // Scale coefficients appropriately for logistic regression
    double scale = 0.5;  // Reduced scale to balance predictions
    
    double *rain_coefficients = (double *)malloc(sizeof(double) * 4);
    rain_coefficients[0] = intercept;
    rain_coefficients[1] = cov_hum * scale;
    rain_coefficients[2] = cov_wind * scale;
    rain_coefficients[3] = cov_pres * scale;

    // Free temporary arrays
    for (int i = 0; i < n; i++)
    {
        free(X[i]);
    }
    free(X);
    free(y);

    return rain_coefficients;
}
