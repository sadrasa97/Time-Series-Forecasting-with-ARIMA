Certainly! Below is a README file for the project, explaining how to run the code and what each part does.

---

# Time Series Forecasting with ARIMA

This project involves using the ARIMA model to forecast the next four steps in a given time series dataset. The dataset is provided in an Excel file (`shkol2.xlsx`). This README provides instructions on how to run the code, as well as a brief explanation of each part of the script.

## Requirements

Ensure you have the following Python packages installed:

- pandas
- statsmodels
- matplotlib
- scikit-learn
- openpyxl

You can install the necessary libraries using pip:

```bash
pip install pandas statsmodels matplotlib scikit-learn openpyxl
```

## Running the Script

1. Place the `shkol2.xlsx` file in the same directory as your script, or update the `file_path` variable to the correct path of the file.

2. Copy and paste the following script into a Python file (e.g., `forecast.py`):

    ```python
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    file_path = 'shkol2.xlsx'
    data = pd.read_excel(file_path)

    data_numeric = data.iloc[0, :-4].dropna().astype(float).reset_index(drop=True)


    model = ARIMA(data_numeric, order=(5,1,0))
    model_fit = model.fit()


    pred_steps = 4
    forecast = model_fit.forecast(steps=pred_steps)

    print("Predicted values for the next four steps:")
    print(forecast)

    extended_data = np.append(data_numeric.values, forecast.values)


    plt.figure(figsize=(12, 6))
    plt.plot(data_numeric, label='Original Data')
    plt.plot(range(len(data_numeric), len(extended_data)), forecast, label='Predicted Data', color='red')
    plt.title('Original Data and Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


    # Since we don't have true future values, let's split the data into train and test sets to evaluate the model
    train_size = int(len(data_numeric) * 0.8)
    train, test = data_numeric[:train_size], data_numeric[train_size:]


    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()


    test_forecast = model_fit.forecast(steps=len(test))


    mae = mean_absolute_error(test, test_forecast)
    rmse = np.sqrt(mean_squared_error(test, test_forecast))

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')


    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Train Data')
    plt.plot(range(train_size, len(data_numeric)), test, label='Test Data')
    plt.plot(range(train_size, len(data_numeric)), test_forecast, label='Forecasted Data', color='red')
    plt.title('Train, Test, and Forecasted Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
    ```

3. Run the script using Python:

    ```bash
    python forecast.py
    ```

## Explanation of the Script

- **Loading the Data**: The script loads the time series data from `shkol2.xlsx`.
- **Data Preprocessing**: The numerical data is extracted and cleaned.
- **ARIMA Model**: The ARIMA model is fitted to the cleaned data.
- **Predictions**: The script forecasts the next four steps and prints the predicted values.
- **Visualization**: The script plots the original data and the predicted values.
- **Evaluation**: The data is split into training and testing sets to evaluate the model's performance using MAE and RMSE. The script then plots the training, testing, and forecasted data.
![Original Data and Predictions](https://github.com/sadrasa97/Time-Series-Forecasting-with-ARIMA/assets/84921331/214223e8-bf10-4a15-97e3-802f6f95b3ec)
![train](https://github.com/sadrasa97/Time-Series-Forecasting-with-ARIMA/assets/84921331/bb9fd76b-aa38-4db8-b98f-11f784bb5269)
