#include <stdio.h>

// Function to calculate linear regression parameters
void linearRegression(double x[], double y[], int n, double *slope, double *intercept) {
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    // Calculate sums
    for (int i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }

    // Calculate slope (m) and intercept (b)
    *slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    *intercept = (sumY - (*slope) * sumX) / n;
}

int main() {
    FILE *file = fopen("data.txt", "r");
    if (file == NULL) {
        printf("Error: Could not open file.\n");
        return 1;
    }

    double x[100], y[100]; // Arrays to store x and y values
    int n = 0;

    // Read data from file
    while (fscanf(file, "%lf %lf", &x[n], &y[n]) == 2) {
        n++;
    }
    fclose(file);

    if (n < 2) {
        printf("Error: Not enough data points.\n");
        return 1;
    }

    double slope, intercept;
    linearRegression(x, y, n, &slope, &intercept);

    printf("Linear Regression Results:\n");
    printf("Slope (m): %.2f\n", slope);
    printf("Intercept (b): %.2f\n", intercept);

    return 0;
}