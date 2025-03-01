#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "Coefficient2.h"

typedef struct
{
    float Size;
    int Location;
    int Security;
    int Garage;
    int Floor_Number;
    int Number_of_Rooms;
    int Built_Year;
    float Price;
} Data;

int loadData(const char *filename, Data **data, int *dataCount)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Error opening file %s\n", filename);
        return -1;
    }

    char line[400];
    int count = 0;

    while (fgets(line, sizeof(line), file))
    {
        count++;
    }
    rewind(file);

    *dataCount = count - 1;
    *data = (Data *)malloc(*dataCount * sizeof(Data));

    fgets(line, sizeof(line), file); // Skip header row
    int index = 0;
    
    while (fgets(line, sizeof(line), file) && index < *dataCount)
    {
        char *token = strtok(line, ",");
        if (token != NULL) {
            (*data)[index].Size = atof(token);
        }

        token = strtok(NULL, ",");
        if (token != NULL) {
            (*data)[index].Location = atoi(token);
        }

        token = strtok(NULL, ",");
        if (token != NULL) {
            (*data)[index].Security = atoi(token);
        }

        token = strtok(NULL, ",");
        if (token != NULL) {
            (*data)[index].Garage = atoi(token);
        }

        token = strtok(NULL, ",");
        if (token != NULL) {
            (*data)[index].Floor_Number = atoi(token);
        }

        token = strtok(NULL, ",");
        if (token != NULL) {
            (*data)[index].Number_of_Rooms = atoi(token);
        }

        token = strtok(NULL, ",");
        if (token != NULL) {
            (*data)[index].Built_Year = atoi(token);
        }
        
        token = strtok(NULL, ",");
        if (token != NULL) {
            (*data)[index].Price = atof(token);
        }

        index++;
    }
    
    fclose(file);
    return 0;
}

void normalizeFeatures(float *data, int size)
{
    float min = data[0], max = data[0];
    for (int i = 1; i < size; i++)
    {
        if (data[i] < min)
            min = data[i];
        if (data[i] > max)
            max = data[i];
    }

    float range = max - min;
    if (range == 0) return; // Avoid division by zero

    for (int i = 0; i < size; i++)
    {
        data[i] = (data[i] - min) / range;
    }
}

// Linear regression for price prediction
float predictPrice(float Size, int Location, int Security, int Garage, int Floor_Number, int Number_of_Rooms, int Built_Year, double *coefficients)
{
    // Coefficients[0] is the intercept
    return coefficients[0] + 
           coefficients[1] * Size + 
           coefficients[2] * Location + 
           coefficients[3] * Security + 
           coefficients[4] * Garage + 
           coefficients[5] * Floor_Number + 
           coefficients[6] * Number_of_Rooms + 
           coefficients[7] * Built_Year;
}

void splitDataset(Data *data, int dataCount, Data **trainSet, Data **testSet, int *trainCount, int *testCount, float ratio)
{
    *trainCount = (int)(dataCount * ratio);
    *testCount = dataCount - *trainCount;

    *trainSet = (Data *)malloc(*trainCount * sizeof(Data));
    *testSet = (Data *)malloc(*testCount * sizeof(Data));

    for (int i = 0; i < *trainCount; i++)
    {
        (*trainSet)[i] = data[i];
    }

    for (int i = *trainCount; i < dataCount; i++)
    {
        (*testSet)[i - *trainCount] = data[i];
    }
}

// Model accuracy - returns percentage of correct predictions
float evaluateModel(Data *testSet, int testCount, double *coefficients)
{
    int correct = 0;
    float errorSum = 0;
    float errorThreshold = 10000; // Define a reasonable error threshold for house prices

    for (int i = 0; i < testCount; i++)
    {
        float Size = testSet[i].Size;
        int Location = testSet[i].Location;
        int Security = testSet[i].Security;
        int Garage = testSet[i].Garage;
        int Floor_Number = testSet[i].Floor_Number;
        int Number_of_Rooms = testSet[i].Number_of_Rooms;
        int Built_Year = testSet[i].Built_Year;

        float pricePrediction = predictPrice(Size, Location, Security, Garage, Floor_Number, Number_of_Rooms, Built_Year, coefficients);

        float error = fabs(pricePrediction - testSet[i].Price);
        errorSum += error;
        
        if (error < errorThreshold)
        {
            correct++;
        }
    }
    
    printf("Average prediction error: %.2f\n", errorSum / testCount);
    return (float)correct / testCount;
}

int main()
{
    Data *data;
    int dataCount;

    if (loadData("house_price_dataset.csv", &data, &dataCount) != 0)
    {
        return -1;
    }

    printf("Dataset loaded with %d records.\n", dataCount);

    // Split
    Data *trainSet, *testSet;
    int trainCount, testCount;
    splitDataset(data, dataCount, &trainSet, &testSet, &trainCount, &testCount, 0.8);

    printf("Training data count: %d, Testing data count: %d\n", trainCount, testCount);

    double *coefficients = finalCoefficientCalc();
    
    printf("\nCoefficients:\n");
    printf("Intercept: %f\n", coefficients[0]);
    printf("Size: %f\n", coefficients[1]);
    printf("Location: %f\n", coefficients[2]);
    printf("Security: %f\n", coefficients[3]);
    printf("Garage: %f\n", coefficients[4]);
    printf("Floor Number: %f\n", coefficients[5]);
    printf("Number of Rooms: %f\n", coefficients[6]);
    printf("Built Year: %f\n", coefficients[7]);

    float accuracy = evaluateModel(testSet, testCount, coefficients);
    printf("Model accuracy: %.2f%%\n", accuracy * 100);

    // Predict price for a given condition
    srand(time(NULL));
    float Size = 1000 + (rand() % 1000);
    int Location = (rand() % 2);  
    int Security =(rand() % 2);
    int Garage = (rand() % 2);
    int Floor_Number = 1 + (rand() % 10);
    int Number_of_Rooms = 1 + (rand() % 10);
    int Built_Year = 2000 + (rand() % 25);

    printf("\nInput Condition:\n");
    printf("Size: %.2f\n", Size);
    printf("Location: %d\n", Location);
    printf("Security: %d\n", Security);
    printf("Garage: %d\n", Garage);
    printf("Floor Number: %d\n", Floor_Number);
    printf("Number of Rooms: %d\n", Number_of_Rooms);
    printf("Built Year: %d\n", Built_Year);  
    printf("\n");

    float predictedPrice = predictPrice(Size, Location, Security, Garage, Floor_Number, Number_of_Rooms, Built_Year, coefficients);
    printf("Predicted House Price: $%.2f\n", predictedPrice);

    // Free allocated memory
    free(coefficients);
    free(data);
    free(trainSet);
    free(testSet);

    return 0;
}