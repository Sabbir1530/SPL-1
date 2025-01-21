#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Define structure to hold weather dataset information
typedef struct {
    char Date[20];
    char Season[10];
    float Temperature;
    float Humidity;
    float Wind_Speed;
    float Pressure;
    int Rain;  // 1 for 'rain', 0 for 'no rain'
} WeatherData;

// Function to load dataset from file
int loadData(const char *filename, WeatherData **data, int *dataCount) {
    FILE *file = fopen("processed_dataset.csv", "r");
    if (!file) {
        printf("Not opening file\n");
        return -1;
    }

    char line[400];
    int count = 0;

    while (fgets(line, sizeof(line), file)) {
        count++;
    }
    rewind(file);

    *dataCount = count - 1;  // excluding header
    *data = (WeatherData *)malloc(*dataCount * sizeof(WeatherData));

    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        if (index == 0) {
            index++;  // skip header line
            continue;
        }

        char *token = strtok(line, ",");
        strcpy((*data)[index - 1].Date, token); 

        token = strtok(NULL, ",");
        strcpy((*data)[index - 1].Season, token); 

        token = strtok(NULL, ",");
        (*data)[index - 1].Temperature = atof(token);  

        token = strtok(NULL, ",");
        (*data)[index - 1].Humidity = atof(token);

        token = strtok(NULL, ",");
        (*data)[index - 1].Wind_Speed = atof(token); 

        token = strtok(NULL, ",");
        (*data)[index - 1].Pressure = atof(token);  

        token = strtok(NULL, ",");
        (*data)[index - 1].Rain = (strcmp(token, "rain\n") == 0) ? 1 : 0; 

        index++;
    }
    fclose(file);
    return 0;
}

// Function to normalize features
void normalizeFeatures(float *data, int size) {
    float min = data[0], max = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }

    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - min) / (max - min);  // Normalize between 0 and 1
    }
}

// Function to perform logistic regression for rain prediction
float predictRain(float humidity, float wind_speed, float pressure, float *coefficients) {
    float features[3] = {humidity, wind_speed, pressure};
    float dotProduct = 0;

    for (int i = 0; i < 3; i++) {
        dotProduct += coefficients[i] * features[i];  // Compute the weighted sum
    }

    float probability = 1 / (1 + exp(-dotProduct));  
    return (probability > 0.5) ? 1 : 0; 
}

// Function to perform linear regression for temperature prediction
float predictTemperature(float humidity, float wind_speed, float pressure, float *coefficients) {
    float features[3] = {humidity, wind_speed, pressure};
    float dotProduct = 0;

    for (int i = 0; i < 3; i++) {
        dotProduct += coefficients[i] * features[i];  // Compute the weighted sum
    }

    return dotProduct;  // Linear regression directly outputs temperature prediction
}

// Function to split dataset into training and testing sets
void splitDataset(WeatherData *data, int dataCount, WeatherData **trainSet, WeatherData **testSet, int *trainCount, int *testCount, float ratio) {
    *trainCount = (int)(dataCount * ratio);
    *testCount = dataCount - *trainCount;

    *trainSet = (WeatherData *)malloc(*trainCount * sizeof(WeatherData));
    *testSet = (WeatherData *)malloc(*testCount * sizeof(WeatherData));

    for (int i = 0; i < *trainCount; i++) {
        (*trainSet)[i] = data[i];  // Copy training data
    }

    for (int i = *trainCount; i < dataCount; i++) {
        (*testSet)[i - *trainCount] = data[i];  // Copy testing data
    }
}

// Function to train the model (rain + temperature regression)
void trainModel(WeatherData *trainSet, int trainCount, float *coefficients) {
    // For simplicity, this example uses fixed coefficients.
    coefficients[0] = 0.5;  // Coefficient for humidity
    coefficients[1] = 0.3;  // Coefficient for wind speed
    coefficients[2] = -0.1;  // Coefficient for pressure
}

// Function to evaluate model accuracy
float evaluateModel(WeatherData *testSet, int testCount, float *coefficients) {
    int correct = 0;
    int tempCorrect = 0;

    for (int i = 0; i < testCount; i++) {
        float hum = testSet[i].Humidity;
        float wind = testSet[i].Wind_Speed;
        float pres = testSet[i].Pressure;

        float rainPrediction = predictRain(hum, wind, pres, coefficients);  // Predict rain
        float tempPrediction = predictTemperature(hum, wind, pres, coefficients);  // Predict temperature

        if ((int)rainPrediction == testSet[i].Rain) {  // Compare predicted rain with actual rain
            correct++;
        }

        if (fabs(tempPrediction - testSet[i].Temperature) < 1.0) {  // Compare predicted temperature with actual temperature
            tempCorrect++;
        }
    }

    printf("Rain Prediction Accuracy: %.2f%%\n", (float)correct / testCount * 100);  // Calculate rain accuracy
    return (float)correct / testCount;
}

// Function to predict temperature and season for a given date
void predictTemperatureAndSeason(const char *date, WeatherData *data, int dataCount) {
    for (int i = 0; i < dataCount; i++) {
        if (strcmp(data[i].Date, date) == 0) {
            printf("Date: %s\n", data[i].Date);
            printf("Predicted Temperature: %.2f\n", data[i].Temperature);
            printf("Predicted Season: %s\n", data[i].Season);
            return;
        }
    }
    printf("Date not found in dataset.\n");
}

int main() {
    WeatherData *data;
    int dataCount;

    if (loadData("processed_dataset.csv", &data, &dataCount) != 0) {
        return -1;
    }

    printf("Dataset loaded with %d records.\n", dataCount);

    // Split the dataset into training and testing sets
    WeatherData *trainSet, *testSet;
    int trainCount, testCount;
    splitDataset(data, dataCount, &trainSet, &testSet, &trainCount, &testCount, 0.8);

    printf("Training data count: %d, Testing data count: %d\n", trainCount, testCount);

    // Train the model (rain + temperature regression)
    float coefficients[3];
    trainModel(trainSet, trainCount, coefficients);

    // Evaluate the model
    float accuracy = evaluateModel(testSet, testCount, coefficients);
    printf("Model accuracy: %.2f%%\n", accuracy * 100);

    // Example of making a prediction for rain
    float inputHumidity = 70.0;
    float inputWindSpeed = 3.0;
    float inputPressure = 1015.0;

    float rainPrediction = predictRain(inputHumidity, inputWindSpeed, inputPressure, coefficients);
    printf("Prediction: %s\n", rainPrediction == 1 ? "Rain" : "No Rain");

    // Predict temperature for a given condition
    float humidity = 65.0;
    float wind_speed = 3.0;
    float pressure = 1012.0;

    float predictedTemp = predictTemperature(humidity, wind_speed, pressure, coefficients);
    printf("Predicted Temperature: %.2f°C\n", predictedTemp);

    // Predict temperature and season for a given date
    char inputDate[20];
    printf("Enter a date (MM/DD/YYYY): ");
    scanf("%19s", inputDate);
    predictTemperatureAndSeason(inputDate, data, dataCount);

    // Free allocated memory
    free(data);
    free(trainSet);
    free(testSet);

    return 0;
}
