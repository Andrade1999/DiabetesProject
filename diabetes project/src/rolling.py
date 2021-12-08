######################################################################ARIMA#######################################################################3
import numpy as np
import pandas as pd
from glob import glob
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from builtins import len
import os
from sklearn.svm._libsvm import predict
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
####################################################
#What does this do?
####################################################
def prepare_data(sequences, lookback, prediction_horizon, validation_split=None): 
    samples = []
    targets = []
    for seq in sequences:
        assert seq.shape[0] > lookback + prediction_horizon
        for i in range(seq.shape[0] - lookback - (prediction_horizon - 1)):
            samples.append(seq[i: i + lookback, :])
            targets.append(seq[i + lookback + prediction_horizon - 1, 0])
    samples = np.stack(samples, axis=0)
    targets = np.stack(targets, axis=0)[:, None]

    if validation_split is not None:
        num_train = int(samples.shape[0] * (1 - validation_split))
        num_val = samples.shape[0] - num_train
    else:
        num_train = samples.shape[0]

    train_samples = samples[0: num_train, :, :]
    train_targets = targets[0: num_train, :]
    if validation_split is not None:
        val_samples = samples[-num_val:, :, :]
        val_targets = targets[-num_val:, :]
        return train_samples, train_targets, val_samples, val_targets
    else:
        return train_samples, train_targets

def get_hypo_event(segment, threshold=70, num_above=3, num_in_hypo=3):
    # TODO: implement a function that returns a binary squence, indicating if we are in a hypo-event (1) or not (0)
    # hint: the CGM values are rescaled to a range of approximately 0 to 1 by dividing them by 400 in "prepare_data". either scale up the
    # segments by multiplying by 400 or divide the threshold by 400 in order to have both in the same range
    threshold = threshold/400
    bsequence = []
    auxstart = 0
    auxend = 0
    i = 1
    bsequence.append(0)
    if segment[0] < threshold:
        auxstart = auxstart + 1            
    while i < np.size(segment): ##change for for with other check condition before
        x = segment[i]
        if bsequence[i-1] == 0:
            if x < threshold:
                auxstart = auxstart + 1
            else:
                auxstart = 0
            if auxstart == num_in_hypo:
                bsequence.append(1)
            else:
                bsequence.append(0)
        elif bsequence[i-1] == 1:
            if x > threshold:
                auxend = auxend + 1
            else:
                auxend = 0
            if auxend == num_above:
                bsequence.append(0)
            else:
                bsequence.append(1)
        i = i + 1
    return np.array(bsequence)

def metrics(gt_events, pred_events):
    assert len(gt_events) == len(pred_events)
    tp_events_candidates = gt_events + pred_events
    tp_events_mask = np.zeros_like(gt_events)
    i = 0
    start_event = None
    while i < len(gt_events):
        if tp_events_candidates[i] > 0:
            # look out if there will come a 2 indicating that there is an overlap between gt and pred
            for j in range(i, len(gt_events), 1):
                if tp_events_candidates[j] == 0:
                    stop_event = j
                    break
                elif j == len(gt_events) - 1:
                    stop_event = len(gt_events)
                elif tp_events_candidates[j] == 1:
                    continue
                elif tp_events_candidates[j] == 2:
                    start_event = i
            if start_event is not None:
                tp_events_mask[start_event:stop_event] = 1
            start_event = None
            i = stop_event
        else:
            i = i + 1
    tp_events = np.convolve(tp_events_mask, np.asarray([1, -1]))
    tp = len(np.where(tp_events == 1)[0])  # count rising edges

    # calculate number of TN
    tn_events = np.where((gt_events + tp_events_mask) > 0, 1, 0)  # add the false negative to the tp mask
    tn_events = np.bitwise_not(tn_events.astype(bool)) * 1  # invert
    tn_events = np.convolve(tn_events, np.asarray([1, -1]), mode='same')  # get rising and falling edges
    tn = len(np.where(tn_events == 1)[0])  # count rising edges

    # calculate number of FP
    fp_events = np.bitwise_and(np.bitwise_not(gt_events.astype(bool)) * 1, pred_events)
    # whenever a predicted hypo event is longer than the ground truth event, we have noise that needs to filtered out
    fp_events = np.bitwise_and(np.bitwise_not(tp_events_mask.astype(bool)) * 1, fp_events)
    fp_events = np.convolve(fp_events, np.asarray([1, -1]))
    fp = len(np.where(fp_events == 1)[0])

    # calculate number of FN
    fn_events = np.bitwise_and(gt_events, np.bitwise_not(pred_events.astype(bool)) * 1)
    fn_events = np.bitwise_and(np.bitwise_not(tp_events_mask.astype(bool)) * 1, fn_events)
    fn_events = np.convolve(fn_events, np.asarray([1, -1]))
    fn = len(np.where(fn_events == 1)[0])

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity

def rmse(x,y):
    error = (x-y)*400 # scale up the output
    squared_error = np.square(error)
    mean_squared_error = np.mean(squared_error)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error

def avg(lst):
    return sum(lst)/len(lst)

if __name__ == "__main__":
    prediction_horizon = 12  # 30 minutes ahead in time
    threshold = 70  # threshold for hypoglycemic events
    
    patient_root = "540-ws-training_processed.csv"  # TODO: path to csv file
    patient_root_test ="540-ws-testing_processed.csv" # TODO: path to csv file
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
    # The sequences contain missing measurements at some time steps due to sensor and/or user errors or off-time.
    # We only select sequences without any interruption for at least half a day (144 5-minute steps = 12h)
    
    #Starting with the first patient [0]
    prediction_horizon = 12
    
    #TODO: implement ARIMA to predict on the testset

    
    #Print of the fit summary
    #print(model_fit.summary())
    
    ######################
    #This checks for bias in the prediction -> what does this mean
    ######################
    #line plot of residuals
    # residuals = pd.DataFrame(model_fit.resid)
    # residuals.plot()
    # pyplot.show()
    # #density plot of residuals
    # residuals.plot(kind='kde')
    # pyplot.show()
    # # summary stats of residuals
    # print(residuals.describe())
    r_sensitivity = []
    r_specificity = []
    r_RMSE = []
    r_performance = []
    os.mkdir("Rolling")
    textfile = open("Rolling/Results_Rolling.txt", "w")
    for x in range(len(testpaths)): #loop for every patient
        predictions = []
        history = train[x]
        for j in range(0, np.size(test[x]), prediction_horizon): #loop for every prediction step
            print(str(j) + "Size: " + str(np.size(test[x])))
            print("Start Model")
            model = ARIMA(history, order = (1, 0, 10)) #fit the model
            model_fit = model.fit()
            #print(model_fit.summary())
            print("Out of Sample Forecast :" + str(x))
            predictions_p = model_fit.forecast(steps = 6) #predict next 6 timesteps
            for z in predictions_p:
                predictions.append(z)
            np.append(history,test[x][j:j + prediction_horizon])
            print(np.size(predictions))

            textfile.write(str(j) + "\n")
            textfile.write("Test " + str(test[x][j:j + prediction_horizon]) + "\n")
            textfile.write("Pred " + str(predictions_p) + "\n\n")
                
        textfile.write("\nResults Patient" + str(x) + "\n")
        targets = test[x]
        print(targets)
        
        print(np.size(targets))
        print(np.size(predictions))

        RMSE_p = sqrt(mean_squared_error(targets, predictions))
        print('Test RMSE P: %.3f' % RMSE_p)
        textfile.write("RSME_P: " + str(RMSE_p) + "\n")

        Performance_p = rmse(targets, predictions)
        print('Test Performance P: %.3f' % Performance_p)
        textfile.write("Performance_P: " + str(Performance_p) + "\n")
        
        os.mkdir("Images_Rolling");
        pyplot.close()
        pyplot.plot(targets)
        pyplot.plot(predictions, color = 'red')
        pyplot.savefig('Images_Rolling/img_' + str(x) + '.png')
        
        gt_event_masks = get_hypo_event(targets, threshold=threshold)
        pred_event_mask = get_hypo_event(predictions, threshold=threshold)
        
        #check if we have a hypo event in the ground truth
        if np.max(gt_event_masks) == 1:
            sensitivity, specificity = metrics(gt_event_masks, pred_event_mask)
            print('sensitivity P: {}\nspecificity P: {}'.format(sensitivity, specificity))
            textfile.write("Sensitivity_P: " + str(sensitivity) + "Specificity_P: " + str(specificity) + "\n")
            r_sensitivity.append(sensitivity)
            r_specificity.append(specificity)
        else:
            print('patient did not have any phase in GT below {}mg/dl'.format(threshold))
            textfile.write("Sensitivity_P: NA, Specificity_P: NA\n")
            r_specificity.append('NA')
            r_sensitivity.append('NA')
        r_RMSE.append(RMSE_p)
        r_performance.append(Performance_p)
    textfile.write("\nSensitivity Array: " + str(r_sensitivity) + "\n")
    textfile.write("Specificity Array: " + str(r_specificity) + "\n")
    textfile.write("RMSE Array: " + str(r_RMSE) + "\n")
    textfile.write("Performance Array: " + str(r_performance) + "\n")
    textfile.write("Average RMSE: " + str(avg(r_RMSE)) + "\nAverage Performance: " + str(avg(r_performance)) + "\n")
    
    
    
    