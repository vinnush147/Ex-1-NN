<H3>NAME : Vinnush Kumar L S</H3>
<H3>REGISTER NO : 212223230244.</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 3/05/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv("/content/Churn_Modelling.csv")
df

df.isnull().sum()


#check for duplication
df.duplicated()

print(df['CreditScore'].describe())

df.info()

df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df

Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

X = df1.iloc[:, :-1].values
print(X)

y = df1.iloc[:,-1].values
print(y)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

print(len(X_train))

print(X_test)

print(len(X_test))
```



## OUTPUT:
<img width="1530" height="441" alt="image" src="https://github.com/user-attachments/assets/caf5518c-8c5f-435d-86d6-cc481396d676" />

<img width="403" height="577" alt="image" src="https://github.com/user-attachments/assets/33ebfee6-39c1-46f1-8ef0-b53a3d827cc2" />

<img width="400" height="529" alt="image" src="https://github.com/user-attachments/assets/6b04238a-42e1-47ca-991e-4c1f875abcb1" />

<img width="540" height="225" alt="image" src="https://github.com/user-attachments/assets/a9ff0411-b76e-4713-b4bd-0c93fe1ba8f7" />

<img width="648" height="422" alt="image" src="https://github.com/user-attachments/assets/f7211094-ddd5-4563-8eae-63e11e2e8a53" />


<img width="1445" height="491" alt="image" src="https://github.com/user-attachments/assets/3729adda-a80b-4fcf-9d14-b06c25bfa976" />

<img width="1018" height="515" alt="image" src="https://github.com/user-attachments/assets/45efe47c-3b08-4cb3-ac51-89ce7fe8bc40" />

<img width="803" height="318" alt="image" src="https://github.com/user-attachments/assets/2588a3e4-4de6-4890-8892-26718d6213bd" />

<img width="363" height="99" alt="image" src="https://github.com/user-attachments/assets/33fb697e-951f-4b48-86cd-9bd5a6017aa2" />

<img width="787" height="249" alt="image" src="https://github.com/user-attachments/assets/0ed346e6-9a5a-45a8-8a3e-f1c4b089b72a" />

<img width="811" height="352" alt="image" src="https://github.com/user-attachments/assets/8f636409-f80a-4585-94c4-b5ec8fe8461f" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


