######################################################################ARIMA#######################################################################
import numpy as np
import pandas as pd
from glob import glob
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from builtins import len
import os

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

def get_hypo_event(segment, threshold=70, num_above=3, num_in_hypo=3):
    #function that returns a binary squence, indicating if we are in a hypo-event (1) or not (0)

    threshold = threshold/400
    bsequence = []
    auxstart = 0
    auxend = 0
    i = 1
    bsequence.append(0)
    if segment[0] < threshold:
        auxstart = auxstart + 1            
    while i < np.size(segment): 
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

#calculate the average of the arrays, ignoring NA for specificity and sensitivity
def avg(lst):
    i = 0;
    aux = 0;
    for x in lst:
        if x != 'NA':
            i = i + 1
            aux = aux + x
    return aux/i

if __name__ == "__main__":
    threshold = 70  # threshold for hypoglycemic events
    
    testpaths = glob('*-ws-testing_processed.csv') #format for test csv
    trainpaths = glob('*-ws-training_processed.csv') #format for train csv
    print(testpaths)
    print(trainpaths)
    train_dataset = []
    test_dataset = []
    train = []
    test = []
    #fill test and train arrays of paths
    for x in range(len(testpaths)):
        train_dataset.append(load_data(csv_file=trainpaths[x]))
        test_dataset.append(load_data(csv_file=testpaths[x]))
        train.append(extract_valid_sequences(train_dataset[x], min_len=144))
        test.append(extract_valid_sequences(test_dataset[x], min_len=144))
    # The sequences contain missing measurements at some time steps due to sensor and/or user errors or off-time.
    # We only select sequences without any interruption for at least half a day (144 5-minute steps = 12h)
    
    #arrays that store metrics for all patients
    r_sensitivity = [] 
    r_specificity = []
    r_RMSE = []
    r_performance = []
    #order of model for each patient has determined in paramater_estimation.py
    order = ((1, 0, 5), (1 , 1, 1), (1, 1, 2), (1, 1, 1), (1, 1, 1), (1, 1, 2), (0, 1, 0), (0, 1, 1), (0, 1, 1), (1, 1, 2), (1, 0, 5), (0, 1, 0))
    #how much do we want to look into the future
    prediction_horizon = 6 #30 minutes
    #create path for results
    if (os.path.isdir("Plot" + str(prediction_horizon)) == False):
        os.mkdir("Plot" + str(prediction_horizon))
       
    #for each patient    
    #for x in range(len(testpaths)):
    x = 4
    odr = order[x] #order of the patient
    lookback = odr[0] #lags we are using as predictors (p parameter)
    textfile = open("Plot" + str(prediction_horizon) + "/Results" + str(x) + ".txt", "w")
    textfile.write("\nResults Patient" + str(x) + "\n")
    print("Start Model")
    history = [i for i in train[x][-200:]] #fill in array for training with last 200 observations (around 17 hours)
    history.extend([i for i in test[x][:lookback]]) #insert the observations we are using has predictors
    test[x] = test[x][lookback:] #remove predictors from test set
    predictions = []
    targets = []
    #for size of test (moves the window forward)
    for i in range(np.size(test[x]) - prediction_horizon):
        incr_history = history.copy()
        #for each observation we want to predict
        for j in range(prediction_horizon):
            incr_predictions = []
            #train the model
            model = ARIMA(incr_history, order = odr)
            #fit the model
            model_fit = model.fit()
            print("Out of Sample Forecast x:" + str(x) + " i:" + str(i) + " j:" + str(j))
            #predict the next observation
            output = model_fit.forecast()
            otp = output[0]
            #add prediction to prediction array and to history
            incr_predictions.append(otp)
            incr_history.append(otp)
            #move the window forward 1
            incr_history = incr_history[1:]
            
        predictions.append(incr_predictions[-1])
        targets.append(test[x][i + prediction_horizon - 1])  
        history.append(test[x][i])
        history = history[1:]
    
    #Root Mean Square Error
    RMSE = sqrt(mean_squared_error(targets*400, predictions*400))
    #Error in mg/dl (as in literature)
    Performance = RMSE*400
    print('Test RMSE P: %.3f' % RMSE)
    #Write results
    textfile.write("RSME: " + str(RMSE) + "\n")
    textfile.write("Performance: " + str(Performance) + "\n")
    
    #Save observations plot
    pyplot.close()
    line1, = pyplot.plot(targets)
    line2, = pyplot.plot(predictions, color = 'red')
    pyplot.xlabel("Time Steps")
    pyplot.ylabel("Normalized Blood Glucose (mg/dl)")
    pyplot.legend([line1, line2], ['Targets', 'Predictions'])
    pyplot.title("Blood glucose prediction vs Training Set")
    pyplot.savefig("Plot" + str(prediction_horizon)  + "/patient_" + str(x) +  '.png')
    
    gt_event_masks = get_hypo_event(targets, threshold=threshold)
    pred_event_mask = get_hypo_event(predictions, threshold=threshold)
    
    #Save event plot
    pyplot.close()
    pyplot.plot(gt_event_masks)
    pyplot.plot(pred_event_mask, color = 'red')
    pyplot.xlabel("Time Steps")
    pyplot.ylabel("Event")
    pyplot.legend()
    pyplot.title("Event prediction vs Training Set")
    pyplot.show()
    pyplot.savefig("Plot" + str(prediction_horizon)  + "/patient_event" + str(x) +  '.png')
    
    #Check if we have a hypo event in the ground truth
    if np.max(gt_event_masks) == 1:
        sensitivity, specificity = metrics(gt_event_masks, pred_event_mask)
        print('sensitivity P: {}\nspecificity P: {}'.format(sensitivity, specificity))
        textfile.write("Sensitivity_P: " + str(sensitivity) + "\nSpecificity_P: " + str(specificity) + "\n")
        r_sensitivity.append(sensitivity)
        r_specificity.append(specificity)
    else:
        print('patient did not have any phase in GT below {}mg/dl'.format(threshold))
        textfile.write("Sensitivity_P: NA, Specificity_P: NA\n")
        r_specificity.append('NA')
        r_sensitivity.append('NA')
        
    textfile.close()
    r_RMSE.append(RMSE)
    r_performance.append(Performance)
        
    #Write final results to a txt
    textfile = open("Plot" + str(prediction_horizon) + "/Results.txt", "w")
    textfile.write("\nSensitivity Array: " + str(r_sensitivity) + "\n")
    textfile.write("Specificity Array: " + str(r_specificity) + "\n")
    textfile.write("RMSE Array: " + str(r_RMSE) + "\n")
    textfile.write("Performance Array: " + str(r_performance) + "\n")
    textfile.write("Average RMSE: " + str(avg(r_RMSE)) + "\nAverage Performance: " + str(avg(r_performance)) + "\n")
    textfile.close()
