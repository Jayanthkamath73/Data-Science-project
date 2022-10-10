import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import random

df = pd.read_csv('C:/Users/User/Documents/Data_Analytics_Project/dataset_diabetes/diabetic_data.csv')
print(df)

print(df.head())

missing_values = ["?", "n/a", "na", "NA", " ", "--", "N"]

df = pd.read_csv('C:/Users/User/Documents/Data_Analytics_Project/dataset_diabetes/diabetic_data.csv', na_values = missing_values)

print("\n")
print(" Number of Rows : ", len(df.axes[0]))
print(" Number of Columns : ", len(df.axes[1]))

print(" Missing Values in each Column : \n")
print(df.isnull().sum())
print(" \n\nTotal Number of Missing Values in Dataset is : ", df.isnull().sum().sum())


df['race'].fillna("AfricanAmerican", inplace=True)
df['payer_code'].fillna("MC", inplace=True)
df['medical_specialty'].fillna("other", inplace=True)
del df['weight']
df.dropna(subset=['diag_1'], inplace=True)
df.dropna(subset=['diag_2'], inplace=True)
df.dropna(subset=['diag_3'], inplace=True)

columns = list(df.axes[1])


print(" \n\n Number of missing values after imputation : ", df.isnull().sum().sum())
print("  Number of Rows : ", len(df.axes[0]))
print("  Number of Columns : ", len(df.axes[1]))

print("\n\n")
print(df)
print("\n\n")


outliers = []
def check_outliers(list_data, col):
   actual_data = list_data
   outlier_for_col = []
   data = np.sort(list_data)
   q1 = np.percentile(data, 25, interpolation = 'midpoint')
   q2 = np.percentile(data, 50, interpolation = 'midpoint')
   q3 = np.percentile(data, 75, interpolation = 'midpoint')
   iqr = q3 - q1
   low_limit = q1 - (1.5*iqr)
   up_limit = q3 + (1.5*iqr)
 
   for x,ind in zip(actual_data,df.index):
    if ((x> up_limit) or (x<low_limit)):
         outliers.append(x)
         outlier_for_col.append(x)
         df.drop(ind, inplace=True)
   print(" Number of Outliers in ",col," column is ",len(outlier_for_col), " with min = ",min(list_data)," and max = ",max(list_data))
   print("\n---------------------------------------------")
   outlier_for_col.clear()
  
def check_outliers_typ(list_data,list_cond,col):
    outlier_for_col = []   
    for i,ind in zip(list_data,df.index):
       if (i not in list_cond):
          outliers.append(i)
          outlier_for_col.append(i)
          df.drop(ind, inplace=True)   
    print(" Number of Outliers in ",col," column is ",len(outlier_for_col), " with min = ",min(list_data)," and max = ",max(list_data))
    print("\n---------------------------------------------")
    outlier_for_col.clear()

check_outliers_typ(df['admission_type_id'],range(1,9),'admission_type_id')
check_outliers_typ(df['discharge_disposition_id'],range(1,30),'discharge_disposition_id')
check_outliers_typ(df['admission_source_id'],range(1,27),'admission_source_id')

for i in range(8,49):
   if (df.dtypes[columns[i]] == np.int64):
      check_outliers(df[columns[i]], columns[i])

total_no_of_outliers = len(outliers)
print(" \n\nTotal number of outliers in dataset : ", total_no_of_outliers)

print("\n")
print(" Number of rows and columns after removing outliers")
print("  Number of Rows : ", len(df.axes[0]))
print("  Number of Columns : ", len(df.axes[1]))

print("\n")
print(" Min and Max values of each column after removing outliers: \n\n")

for i in columns:
   if (df.dtypes[i] == np.int64):
       print( "Min and max value of ",i," columns are : ",min(df[i])," and ",max(df[i]))

inc_list = []

def check_inc(list_data, list_cond,col):
    count_inc = 0
    for i,ind in zip(list_data,df.index):
       if (i not in list_cond):
          df.drop(ind, inplace=True)
          count_inc += 1
    print(" \n Number of inconsistent values in ",col," column is :",count_inc)
    inc_list.append(count_inc)

def check(list_data,col):
   count_inc = 0
   for i,ind in zip(list_data,df.index):
      if (i < 0 or isinstance(i, float)):
         df.drop(ind, inplace=True)
         count_inc += 1
   print(" \n Number of inconsistent values in ",col," column is :",count_inc)
   inc_list.append(count_inc)

def check_diag(list_data,col):
   count_diaginc = 0
   for i,ind in zip(list_data,df.index):
      if (i.isalnum()): 
         continue
      elif (i.find(".") != -1 or i.isdigit()):
         continue
      else:
         df.drop(ind, inplace=True)
         count_diaginc += 1
   print(" \n Number of inconsistent values in ",col," column is :",count_diaginc)
   inc_list.append(count_diaginc)


check(df['encounter_id'],'encounter_id')
check(df['patient_nbr'],'patient_nbr')
check_inc(df['gender'],["Female", "female", "Male", "male"],'gender')
check_inc(df['age'],["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"],'age')
check_inc(df['admission_type_id'],range(1,10),'admission_type_id')
check_inc(df['discharge_disposition_id'],range(1,30),'discharge_disposition_id')
check_inc(df['admission_source_id'],range(1,27),'admission_source_id')
check(df['time_in_hospital'],'time_in_hospital')
check(df['num_lab_procedures'],'num_lab_procedures')
check(df['num_procedures'],'num_procedures')
check(df['num_medications'],'num_medications')
check(df['number_outpatient'],'number_outpatient')
check(df['number_emergency'],'number_emergency')
check(df['number_inpatient'],'number_inpatient')
check_diag(df['diag_1'],'diag_1')
check_diag(df['diag_2'],'diag_2')
check_diag(df['diag_3'],'diag_3')
check(df['number_diagnoses'],'number_diagnoses')
check_inc(df['max_glu_serum'],[">200",">300","None","none","Normal","normal","Norm","norm"],'max_glu_serum')
check_inc(df['A1Cresult'],[">7",">8","None","Normal","none","normal","Norm","norm"],'A1Cresult')

columns = list(df.axes[1])
for i in range(23,46):
    check_inc(df[columns[i]],["No","Up","Down","Steady"],columns[i])

check_inc(df['change'],["Ch","No","ch","change","no change"],'change')
check_inc(df['diabetesMed'],["Yes","No","yes","no"],'diabetesMed')
check_inc(df['readmitted'],["NO",">30","<30"],'readmitted')



total_no_of_inc = sum(inc_list)
print(" \n\nTotal number of inconsistent values in dataset : ", total_no_of_inc)



print("\n")
print(" Number of rows and columns after removing inconsistent values")
print(" Number of Rows : ", len(df.axes[0]))
print(" Number of Columns : ", len(df.axes[1]))


readm = ["<30",">30","NO"]
no_of_patients = [0,0,0]

for i in df['readmitted']:
   if (i == "<30"):
      no_of_patients[0] += 1
   elif (i == ">30"):
      no_of_patients[1] += 1
   else:
      no_of_patients[2] += 1

y_pos = np.arange(len(readm))
plt.bar(y_pos,no_of_patients,color=(0.5,0.1,0.5,0.6))
plt.xlabel('Readmission')
plt.ylabel('No of Patients')
plt.xticks(y_pos,readm,rotation='vertical')
plt.show(block=False)
plt.pause(3)
plt.close("all")

group_name = ["Circulatory","Respiratory","Digestive","Diabetes","Injury","Musculoskeletal","Genitourinary,","Neoplasms","Other"]
no_of_enc = [0,0,0,0,0,0,0,0,0]
for i in df['diag_1']:
    if (i.isnumeric()):
       a = int(i)
       if(a in range(390,460) or a==785):
          no_of_enc[0] += 1
       elif (a in range(460,520) or a==786):
          no_of_enc[1] += 1
       elif (a in range(520,580) or a==787):
          no_of_enc[2] += 1
       elif (a in range(800,1000)):
          no_of_enc[4] += 1
       elif (a in range(710,740)):
          no_of_enc[5] += 1
       elif (a in range(580,630) or a==788):
          no_of_enc[6] += 1
       elif (a in range(140,240)):
          no_of_enc[7] += 1
       else:
          no_of_enc[8] += 1
    elif (i[0]=='2' and i[1]=='5' and i[2]=='0'):
       no_of_enc[3] += 1
    else:
       no_of_enc[8] += 1

y_pos = np.arange(len(group_name))
plt.bar(y_pos,no_of_enc,color=(0.5,0.1,0.5,0.6))
plt.xlabel('Group Name')
plt.ylabel('No of Encounters')
plt.xticks(y_pos,group_name,rotation='vertical')
plt.show(block=False)
plt.pause(3)
plt.close("all")

fig = plt.figure(figsize=(10, 7))
plt.pie(no_of_enc, labels = group_name)
plt.show(block=False)
plt.pause(3)
plt.close("all")


cor_list = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient']

data = {
   'time_in_hospital': df['time_in_hospital'],
   'num_lab_procedures': df['num_lab_procedures'],
   'num_procedures': df['num_procedures'],
   'num_medications': df['num_medications'],
   'number_outpatient': df['number_outpatient'],
   'number_emergency': df['number_emergency'],
   'number_inpatient': df['number_inpatient']
}
df_new = pd.DataFrame(data,columns = cor_list)
corr_matrix = df_new.corr()

print("\n")
print(" Correlation Matrix is : \n")
print(corr_matrix)

def cor_plot(x,y):
   plt.scatter(df[x],df[y])
   plt.xlabel(x)
   plt.ylabel(y)
   plt.show(block=False)
   plt.pause(0.5)
   plt.close("all")

for i in cor_list:
   for j in cor_list:
      if (i != j):
         cor_plot(i,j)


readmitted = []

for i in df['readmitted']:
   if (i == ">30"):
      readmitted.append(1)
   elif (i == "<30"):
      readmitted.append(2)
   else:
      readmitted.append(3)

for i in range(0,20):
   print(readmitted[i])
num_data = {
   'time_in_hospital': df['time_in_hospital'],
   'num_lab_procedures': df['num_lab_procedures'],
   'num_procedures': df['num_procedures'],
   'num_medications': df['num_medications'],
   'number_outpatient': df['number_outpatient'],
   'number_emergency': df['number_emergency'],
   'number_inpatient': df['number_inpatient'],
   'readmitted': readmitted
}

num_list = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','readmitted']
dataset = pd.DataFrame(num_data,columns=num_list)

x = dataset.iloc[:, 0:7]
y = dataset.iloc[:, 7]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pca = PCA(n_components = 2)
 
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
 
explained_variance = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)



x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                     stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1,
                     stop = x_set[:, 1].max() + 1, step = 0.01))
 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))
 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
 
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
 
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend
 
# show scatter plot
plt.show(block=False)
plt.pause(3)
plt.close("all")



x_set, y_set = x_test, y_test
 
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                     stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1,
                     stop = x_set[:, 1].max() + 1, step = 0.01))
 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))
 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
 
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
 
# title for scatter plot
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend()
 
# show scatter plot
plt.show(block=False)
plt.pause(3)
plt.close("all")


