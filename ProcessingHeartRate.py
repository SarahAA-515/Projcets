from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

My_DataFrame1 = pd.read_csv("Data/Raw_Data/sensor_hrv.csv")

My_DataFrame2 = pd.read_csv("Data/Raw_Data/survey.csv")

print(My_DataFrame1['missingness_score'].describe())

# The baseline of 0.67 was chosen based on Descriptive statistics to ensure higher quality removing high-noise readings. 

# Person A is the basline.
Person_A = My_DataFrame1[(My_DataFrame1['deviceId'] == 'pm96') & (My_DataFrame1['missingness_score'] < 0.67)]

HR_val = Person_A['HR']

HRhis = sns.histplot(data = Person_A, x = 'HR')

HRhis.set_xlabel("Heart Rate (bpm)")

HRhis.set_ylabel("Frequency")

HRhis.set_title("Person_A Data")

plt.show()

# nu was chosen based on diagram allowing 10% anomalies.

# gamma was chosen to be set to scale in order to adapt to data spread for a balanced preformance.

svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)

print(svm)

# reshape convert an array into a 2D array with exactly one column and as many rows as required.

svm.fit(Person_A['HR'].values.reshape(-1,1))

pred = svm.predict(Person_A['HR'].values.reshape(-1,1))

anom_index = where(pred==-1)[0]

values = HR_val.values[anom_index]

plt.scatter(range(len(Person_A)),HR_val)

plt.scatter(anom_index, values, color='r')

plt.show()

# person B-F are test set 

Person_B = My_DataFrame1[(My_DataFrame1['deviceId'] == 'ab60') & (My_DataFrame1['missingness_score'] < 0.67)]

BHR_val = Person_B['HR'].values

# size was valued at 10 to simulate a brief touch of a few momnets in real life.

# replace False to prevent the sampling of the same reading twice.

random_sample = np.random.choice(BHR_val, size=10, replace=False)

pred_B = svm.predict(Person_B['HR'].values.reshape(-1,1))

pct_B = (sum(pred_B == -1) / len(pred_B)) * 100 # This counts how many anomalies flagged.

print("Person B:", pct_B, "% flagged")

Person_C = My_DataFrame1[(My_DataFrame1['deviceId'] == 'am77') & (My_DataFrame1['missingness_score'] < 0.67)]

CHR_val = Person_C['HR'].values

random_sample = np.random.choice(CHR_val, size=10, replace=False)

pred_C = svm.predict(Person_C['HR'].values.reshape(-1,1))

pct_C = (sum(pred_C == -1) / len(pred_C)) * 100

print("Person C:", pct_C, "% flagged")

Person_D = My_DataFrame1[(My_DataFrame1['deviceId'] == 'av54') & (My_DataFrame1['missingness_score'] < 0.67)]

DHR_val = Person_D['HR'].values

random_sample = np.random.choice(DHR_val, size=10, replace=False)

pred_D = svm.predict(Person_D['HR'].values.reshape(-1,1))

pct_D = (sum(pred_D == -1) / len(pred_D)) * 100

print("Person D:", pct_D, "% flagged")

Person_F = My_DataFrame1[(My_DataFrame1['deviceId'] == 'ba30') & (My_DataFrame1['missingness_score'] < 0.67)]

FHR_val = Person_F['HR'].values

random_sample = np.random.choice(FHR_val, size=10, replace=False)

pred_F = svm.predict(Person_F['HR'].values.reshape(-1,1))

pct_F= (sum(pred_F == -1) / len(pred_F)) * 100

print("Person F:", pct_F, "% flagged") 
