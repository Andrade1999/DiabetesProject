######################################################################ARIMA#######################################################################3
import numpy as np
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.sandbox.nonparametric.tests.ex_gam_am_new import order

def load_data(csv_file):
    '''
    csv_file: file path of patient's data
    '''
    data = pd.read_csv(csv_file)
    vars_to_include = []
    cbg = data['cbg'].values
    cbg = np.round(cbg) / 400  # resize all samples, so that they lay in range 0.1 to 1, approximately
    vars_to_include.append(cbg)
    vars_to_include.append(data['missing_cbg'].values)
    dataset = np.stack(vars_to_include, axis=1)
    dataset[np.isnan(dataset)] = 0
    return dataset

def extract_valid_sequences(data, min_len=144):
    ValidData = []
    i = 0
    sequence = []
    while i < data.shape[0]:  # dataset.shape[0] = number of train/test samples for one patient
        if data[i, -1] == 1:  # if we have missing values in the cbg measurements
            if len(sequence) > 0:
                if len(sequence) >= min_len:
                    ValidData.append(np.stack(sequence))
                sequence = []
            i = i + 1
        else:
            sequence.append(data[i, :-1])  # do not add the "missing_cbg" column
            i = i + 1
    return np.squeeze(np.concatenate(ValidData, axis=0))

if __name__ == "__main__":
    
    patient_root = "552-ws-training_processed.csv"  # TODO: path to csv file
    patient_root_test ="552-ws-testing_processed.csv" # TODO: path to csv file

    train_dataset = load_data(csv_file=patient_root)
    test_dataset = load_data(csv_file=patient_root_test)
    
    
    # The sequences contain missing measurements at some time steps due to sensor and/or user errors or off-time.
    # We only select sequences without any interruption for at least half a day (144 5-minute steps = 12h)
    train = extract_valid_sequences(train_dataset, min_len=144)
    test = extract_valid_sequences(test_dataset, min_len=144)
    #Check if stationary (p value > 0,05)
    statresulttrain = adfuller(train)
    print((statresulttrain[0]))
    print((statresulttrain[1]))
    
    statresulttest = adfuller(test)
    print((statresulttest[0]))
    print((statresulttest[1]))
    
    pyplot.plot(train)
    pyplot.show()
    
    pyplot.plot(test)
    pyplot.show()
    
    print(train)
    print(test)
    
    #Partial Auto correlation plot, check for p
    plot_pacf(train)
    pyplot.show()
        
    #Check for q
    pd.plotting.autocorrelation_plot(train)
    pyplot.show()
    
    plot_acf(train)
    pyplot.show()
