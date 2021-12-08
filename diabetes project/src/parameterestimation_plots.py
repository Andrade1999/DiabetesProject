######################################################################ARIMA#######################################################################3
import numpy as np
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from glob import glob
import os
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
    
    testpaths = glob('*-ws-testing_processed.csv')
    trainpaths = glob('*-ws-training_processed.csv')
    print(testpaths)
    print(trainpaths)
    train_dataset = []
    test_dataset = []
    train = []
    test = []
    for x in range(len(testpaths)):
        train_dataset.append(load_data(csv_file=trainpaths[x]))
        test_dataset.append(load_data(csv_file=testpaths[x]))
        train.append(extract_valid_sequences(train_dataset[x], min_len=144))
        test.append(extract_valid_sequences(test_dataset[x], min_len=144))
    os.mkdir("Parameters")
    textfile = open("Parameters/Parameters.txt", "w")
    for x in range(len(testpaths)):
        textfile.write("\nPatient: " + str(x) + "\n")
    #Check if stationary (p value > 0,05)
        statresulttrain = adfuller(train[x])
        print((statresulttrain[0]))
        print((statresulttrain[1]))
        textfile.write("Dif Train 1 :" + str(statresulttrain[0]) + "Dif Train 2: " + str(statresulttrain[1]) + "\n")
        
        statresulttest = adfuller(test[x])
        print((statresulttest[0]))
        print((statresulttest[1]))
        textfile.write("Dif Test 1 :" + str(statresulttest[0]) + "Dif Test 2: " + str(statresulttest[1]) + "\n")
        
        pyplot.close()
        pyplot.plot(train[x])
        pyplot.savefig("Parameters/patient_" + str(x) + "train.jpg")
        
        pyplot.close()
        pyplot.plot(test[x])
        pyplot.savefig("Parameters/patient_" + str(x) + "test.jpg")

        #Partial Auto correlation plot, check for p
        pyplot.close()
        plot_pacf(train[x])
        pyplot.savefig("Parameters/patient_" + str(x) + "pacf.jpg")

        #Check for q
        pyplot.close()
        plot_acf(train[x])
        pyplot.savefig("Parameters/patient_" + str(x) + "acf.jpg")
