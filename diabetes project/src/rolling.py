######################################################################ARIMA#######################################################################3
import numpy as np
import pandas as pd
from glob import glob
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from builtins import len
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

if __name__ == "__main__":
    
    prediction_horizon = 6  # 30 minutes ahead in time
    threshold = 70  # threshold for hypoglycemic events
    
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
    history_f = train[0]
    history_p = train[0]
    predictions_f = []
    predictions_p = []
    order = (1,0,0)
    print(len(test[0]))
    for x in range(len(test[0])):
        print(x)
        model_f = ARIMA(history_f, order=order)
        model_p = ARIMA(history_p, order=order)
        model_fit_f = model_f.fit()
        model_fit_p = model_p.fit()
        yhat_f = model_fit_f.forecast()[0]
        yhat_p = model_fit_p.predict(start=len(history_p), end=len(history_p))[0]
        predictions_f.append(yhat_f)
        predictions_p.append(yhat_p)
        np.append(history_f,yhat_f)
        np.append(history_p, yhat_p)
        
    targets = test[0]
    print(np.size(predictions_f))
    print(np.size(targets))
    print(np.size(predictions_p))

    RMSE = sqrt(mean_squared_error(targets, predictions_f))
    print('Test RMSE F: %.3f' % RMSE)
    Performance = rmse(targets, predictions_f)
    print('Test Performance F: %.3f' % Performance)
    
    
    RMSE = sqrt(mean_squared_error(targets, predictions_p))
    print('Test RMSE P: %.3f' % RMSE)
    Performance = rmse(targets, predictions_p)
    print('Test Performance P: %.3f' % Performance)
    
    pyplot.plot(targets)
    pyplot.plot(predictions_f, color = 'red')
    pyplot.plot(predictions_p, color = 'green')
    pyplot.show()
    
    gt_event_masks = get_hypo_event(targets, threshold=threshold)
    pred_event_mask = get_hypo_event(predictions_f, threshold=threshold)
    print(gt_event_masks)
    print(pred_event_mask)
    #check if we have a hypo event in the ground truth
    if np.max(gt_event_masks) == 1:
        sensitivity, specificity = metrics(gt_event_masks, pred_event_mask)
        print('sensitivity F: {}\nspecificity F: {}'.format(sensitivity, specificity))
    else:
        print('patient did not have any phase in GT below {}mg/dl'.format(threshold))
        
    gt_event_masks = get_hypo_event(targets, threshold=threshold)
    pred_event_mask = get_hypo_event(predictions_p, threshold=threshold)
    print(gt_event_masks)
    print(pred_event_mask)
    #check if we have a hypo event in the ground truth
    if np.max(gt_event_masks) == 1:
        sensitivity, specificity = metrics(gt_event_masks, pred_event_mask)
        print('sensitivity P: {}\nspecificity P: {}'.format(sensitivity, specificity))
    else:
        print('patient did not have any phase in GT below {}mg/dl'.format(threshold))
    
    
    
    