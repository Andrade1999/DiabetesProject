'''
Created on 18/11/2021

@author: Ricardo
'''
import io
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.tests.window.conftest import raw
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from PyQt5.Qt import Qt

if __name__ == '__main__':
    raw = io.StringIO("""date        count
2001-11-01   0.998543
2001-11-02   1.914526
2001-11-03   3.057407
2001-11-04   4.044301
2001-11-05   4.952441
2001-11-06   6.002932
2001-11-07   6.930134
2001-11-08   8.011137
2001-11-09   9.040393
2001-11-10  10.097007
2001-11-11  11.063742
2001-11-12  12.051951
2001-11-13  13.062637
2001-11-14  14.086016
2001-11-15  15.096826
2001-11-16  15.944886
2001-11-17  17.027107
2001-11-18  17.930240
2001-11-19  18.984202
2001-11-20  19.971603""")
    matplotlib.use('QtAgg')
    data = pd.read_fwf(raw, parse_dates=['date'], index_col='date')
    datat = []
    x = [1, 2, 3]
    x = np.array(x)
    x = np.vstack([x, [4,5,6]])
    print(x)
    
