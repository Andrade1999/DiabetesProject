# How to Run the Code
There are 2 scripts, one that calculates, for each patient, their parameters for the ARIMA model (p, d, q), based on statistical tests, and one with the ARIMA model.
# Parameter Estimation Script
Run this scrips first. 
The results are stored in a folder called "Parameters", wich contains a .txt file with the ADFuller test results (for parameter d) and the order for each patient. You can analyze both the test value and 
p-value and keep or choose another differencing order.
Inside the same folder you can find the plots corresponding to the Partial Autocorrelation (for parameter p) and Autocorrelation (for parameter q). The code automatically chooses the lags with have correlation values higher than
0.5 but you can analyze these plots and determine if you wanna consider more or less lags according to their correlation value and if they are within 95% confidence level (blue area of the plot).
<br /> You also have access to plots of each training and test set.
# ARIMA Model Script
The main script trains the model and predicts, utilizing a moving window method, a number of observations equal to the variable "prediction_horizon".
You should modify in the script the order of the model of each patient according to what you want to test. The results of this script are stored in a "Results" folder, wich contains
a .txt file for each patient with their RMSE, performance (RMSE times 400, in mg/dl), sensitiviy and specificity. You also have a .txt file with the arrays with all of the RMSE,
performance, sensitivy and specificity results plus the average of each metric. <br />
WARNING: The script for all the patients takes a long time to run (5 or 6 hours). This is mostly due to the patients with big p and q values, with increase the running time of the 
code a lot. This is because their timeseries are not differenced.
