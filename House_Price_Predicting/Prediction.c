#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "Coefficient2.h"

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

    // Count total lines
    while (fgets(line, sizeof(line), file))
    {
        count++;
    }
    
    // Subtract 1 for header row
    *dataCount = count - 1;
    
    if (*dataCount <= 0)
    {
        printf("Error: No data records found in file\n");
        fclose(file);
        return -1;
    }
    
    *data = (Data *)malloc(*dataCount * sizeof(Data));
    
    // Reset file pointer to beginning
    rewind(file);
    
    // Skip header row
    fgets(line, sizeof(line), file);
    
    // Read data records
    int index = 0;
    while (fgets(line, sizeof(line), file) && index < *dataCount)
    {
        // Remove trailing newline if present
        char *newline = strchr(line, '\n');
        if (newline) *newline = '\0';
        
        // Skip empty lines
        if (strlen(line) == 0) continue;
        
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
    
    // Adjust dataCount to actual records loaded
    *dataCount = index;
    
    fclose(file);
    return 0;
}

// Shuffle data for random train/test split
void shuffleData(Data *data, int size)
{
    for (int i = size - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        
        // Swap
        Data temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

// Get normalization parameters from data
NormalizationParams getNormalizationParams(float *values, int size)
{
    NormalizationParams params;
    params.min = values[0];
    params.max = values[0];
    
    for (int i = 1; i < size; i++)
    {
        if (values[i] < params.min)
            params.min = values[i];
        if (values[i] > params.max)
            params.max = values[i];
    }
    
    return params;
}

// Normalize a single value using given parameters (kept for potential future use)
float normalizeValue(float value, NormalizationParams params)
{
    float range = params.max - params.min;
    if (range == 0) return 0;
    return (value - params.min) / range;
}

// Linear regression for raw data price prediction
float predictPrice(float Size, int Location, int Security, int Garage, int Floor_Number, int Number_of_Rooms,
int Built_Year, double *coefficients)
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
    // Shuffle data for random split
    shuffleData(data, dataCount);
    
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

// Calculate accuracy percentage (predictions within acceptable error margin)
float calculateAccuracy(Data *testSet, int testCount, double *coefficients, float errorThreshold)
{
    int correctPredictions = 0;

    for (int i = 0; i < testCount; i++)
    {
        float pricePrediction = predictPrice(testSet[i].Size, testSet[i].Location, testSet[i].Security, 
                                            testSet[i].Garage, testSet[i].Floor_Number, testSet[i].Number_of_Rooms, 
                                            testSet[i].Built_Year, coefficients);
        
        float percentError = fabs((pricePrediction - testSet[i].Price) / testSet[i].Price) * 100;
        
        if (percentError <= errorThreshold)
        {
            correctPredictions++;
        }
    }
    
    return ((float)correctPredictions / testCount) * 100;
}

// Calculate Mean Absolute Error (MAE)
float calculateMAE(Data *testSet, int testCount, double *coefficients)
{
    float errorSum = 0;

    for (int i = 0; i < testCount; i++)
    {
        float pricePrediction = predictPrice(testSet[i].Size, testSet[i].Location, testSet[i].Security, 
                                            testSet[i].Garage, testSet[i].Floor_Number, testSet[i].Number_of_Rooms, 
                                            testSet[i].Built_Year, coefficients);
        
        float error = fabs(pricePrediction - testSet[i].Price);
        errorSum += error;
    }
    
    return errorSum / testCount;
}

// Calculate Root Mean Squared Error (RMSE)
float calculateRMSE(Data *testSet, int testCount, double *coefficients)
{
    float sumSquaredError = 0;

    for (int i = 0; i < testCount; i++)
    {
        float pricePrediction = predictPrice(testSet[i].Size, testSet[i].Location, testSet[i].Security, 
                                            testSet[i].Garage, testSet[i].Floor_Number, testSet[i].Number_of_Rooms, 
                                            testSet[i].Built_Year, coefficients);
        
        float error = pricePrediction - testSet[i].Price;
        sumSquaredError += error * error;
    }
    
    return sqrt(sumSquaredError / testCount);
}

// Calculate Mean Absolute Percentage Error (MAPE)
float calculateMAPE(Data *testSet, int testCount, double *coefficients)
{
    float percentageErrorSum = 0;

    for (int i = 0; i < testCount; i++)
    {
        float pricePrediction = predictPrice(testSet[i].Size, testSet[i].Location, testSet[i].Security, 
                                            testSet[i].Garage, testSet[i].Floor_Number, testSet[i].Number_of_Rooms, 
                                            testSet[i].Built_Year, coefficients);
        
        float error = fabs((pricePrediction - testSet[i].Price) / testSet[i].Price);
        percentageErrorSum += error;
    }
    
    return (percentageErrorSum / testCount) * 100;
}

// Calculate R-squared (Coefficient of Determination)
float calculateRSquared(Data *testSet, int testCount, double *coefficients)
{
    float meanPrice = 0;
    float sumSquaredResidual = 0;
    float sumSquaredTotal = 0;

    // Calculate mean of actual prices
    for (int i = 0; i < testCount; i++)
    {
        meanPrice += testSet[i].Price;
    }
    meanPrice /= testCount;

    // Calculate errors
    for (int i = 0; i < testCount; i++)
    {
        float pricePrediction = predictPrice(testSet[i].Size, testSet[i].Location, testSet[i].Security, 
                                            testSet[i].Garage, testSet[i].Floor_Number, testSet[i].Number_of_Rooms, 
                                            testSet[i].Built_Year, coefficients);
        
        float residual = testSet[i].Price - pricePrediction;
        float total = testSet[i].Price - meanPrice;
        
        sumSquaredResidual += residual * residual;
        sumSquaredTotal += total * total;
    }
    
    if (sumSquaredTotal == 0) return 0;
    return 1 - (sumSquaredResidual / sumSquaredTotal);
}

// Comprehensive model evaluation - displays accuracy
void evaluateModel(Data *testSet, int testCount, double *coefficients)
{
    float rSquared = calculateRSquared(testSet, testCount, coefficients);
    float accuracy5 = calculateAccuracy(testSet, testCount, coefficients, 5.0);   // 5% error threshold
    float accuracy10 = calculateAccuracy(testSet, testCount, coefficients, 10.0); // 10% error threshold
    float accuracy15 = calculateAccuracy(testSet, testCount, coefficients, 15.0); // 15% error threshold
    
    printf("\n========== MODEL ACCURACY METRICS ==========\n");
    printf("Overall R² Accuracy (explained variance):   %.2f%%\n", rSquared * 100);
    printf("Prediction Accuracy (within 5%% error):    %.2f%%\n", accuracy5);
    printf("Prediction Accuracy (within 10%% error):   %.2f%%\n", accuracy10);
    printf("Prediction Accuracy (within 15%% error):   %.2f%%\n", accuracy15);
    printf("===========================================\n");
    
    if (rSquared > 0.9) {
        printf("✓ Model Quality: EXCELLENT (R² > 0.9)\n");
    } else if (rSquared > 0.8) {
        printf("✓ Model Quality: VERY GOOD (R² > 0.8)\n");
    } else if (rSquared > 0.7) {
        printf("✓ Model Quality: GOOD (R² > 0.7)\n");
    } else if (rSquared > 0.6) {
        printf("✓ Model Quality: FAIR (R² > 0.6)\n");
    } else {
        printf("✗ Model Quality: POOR (R² < 0.6)\n");
    }
}

int main()
{
    srand((unsigned int)time(NULL));
    
    Data *data;
    int dataCount;

    if (loadData("house_price_dataset.csv", &data, &dataCount) != 0)
    {
        return -1;
    }

    printf("Dataset loaded with %d records.\n", dataCount);

    // Split data with random shuffle
    Data *trainSet, *testSet;
    int trainCount, testCount;
    splitDataset(data, dataCount, &trainSet, &testSet, &trainCount, &testCount, 0.8);

    printf("Training data count: %d, Testing data count: %d\n\n", trainCount, testCount);

    // Create dummy normalization parameters (not really used anymore, just for interface compatibility)
    NormalizationParams normParams[8];
    for (int i = 0; i < 8; i++) {
        normParams[i].min = 0;
        normParams[i].max = 1;
    }
    
    // Calculate coefficients using raw training data
    double *coefficients = calculateCoefficientsOnTrainingData(trainSet, trainCount, normParams);
    
    printf("========== CALCULATED COEFFICIENTS ==========\n");
    printf("Intercept:        %.6f\n", coefficients[0]);
    printf("Size:             %.6f\n", coefficients[1]);
    printf("Location:         %.6f\n", coefficients[2]);
    printf("Security:         %.6f\n", coefficients[3]);
    printf("Garage:           %.6f\n", coefficients[4]);
    printf("Floor Number:     %.6f\n", coefficients[5]);
    printf("Number of Rooms:  %.6f\n", coefficients[6]);
    printf("Built Year:       %.6f\n", coefficients[7]);
    printf("=============================================\n\n");

    // Evaluate model on TEST data
    evaluateModel(testSet, testCount, coefficients);

    // Test predictions on random houses
    printf("\n========== TEST PREDICTIONS ==========\n");
    for (int test = 0; test < 3; test++)
    {
        int randomIdx = rand() % testCount;
        Data testHouse = testSet[randomIdx];
        
        float pricePrediction = predictPrice(testHouse.Size, testHouse.Location, testHouse.Security, 
                                            testHouse.Garage, testHouse.Floor_Number, testHouse.Number_of_Rooms, 
                                            testHouse.Built_Year, coefficients);
        
        printf("\nTest House %d:\n", test + 1);
        printf("  Size: %.2f sq ft\n", testHouse.Size);
        printf("  Location: %d\n", testHouse.Location);
        printf("  Security: %d\n", testHouse.Security);
        printf("  Garage: %d\n", testHouse.Garage);
        printf("  Floor Number: %d\n", testHouse.Floor_Number);
        printf("  Number of Rooms: %d\n", testHouse.Number_of_Rooms);
        printf("  Built Year: %d\n", testHouse.Built_Year);
        printf("  Actual Price: $%.2f thousand\n", testHouse.Price);
        printf("  Predicted Price: $%.2f thousand\n", pricePrediction);
        printf("  Error: $%.2f thousand (%.2f%%)\n", 
               fabs(pricePrediction - testHouse.Price),
               (fabs(pricePrediction - testHouse.Price) / testHouse.Price) * 100);
    }
    printf("======================================\n");

    // Free allocated memory
    free(coefficients);
    free(data);
    free(trainSet);
    free(testSet);

    return 0;
}
