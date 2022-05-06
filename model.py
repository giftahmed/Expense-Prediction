
################################Step 1 - Import data############################################
import pandas as pd
MyData = pd.read_csv("Income_Expense_Data.csv")



#Checking Size of data
MyData.shape




#Checking first few records
MyData.head(10)




################Step 2-Data Cleaning#######################
#Check for missing values
MyData.isnull().sum() 




#Treating null value-replacing null value with median
MyData["Income"].fillna((MyData["Income"].median()), inplace = True)




#Check for missing values - Again
MyData.isnull().sum() 




#Checking for outliers
MyData.describe()  #notice the maximum value in Age




#Checking different percentiles
pd.DataFrame(MyData['Age']).describe(percentiles=(1,0.99,0.9,0.75,0.5,0.3,0.1,0.01))




#checking boxplot for Age column
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(MyData['Age'])
plt.show()




#Checking Outlier by definition and treating outliers

#getting median Age
Age_col_df = pd.DataFrame(MyData['Age'])
Age_median = Age_col_df.median()

#getting IQR of Age column
Q3 = Age_col_df.quantile(q=0.75)
Q1 = Age_col_df.quantile(q=0.25)
IQR = Q3-Q1

#Deriving boundaries of Outliers
IQR_LL = int(Q1 - 1.5*IQR)
IQR_UL = int(Q3 + 1.5*IQR)

#Finding and treating outliers - both lower and upper end
MyData.loc[MyData['Age']>IQR_UL , 'Age'] = int(Age_col_df.quantile(q=0.99))
MyData.loc[MyData['Age']<IQR_LL , 'Age'] = int(Age_col_df.quantile(q=0.01))




#Check max age value now
max(MyData['Age'])




################Step 3-Exploratory data analysis#######################
#Check how Expense is varying with income
x = MyData["Income"]
y=  MyData["Expense"]


plt.scatter(x, y, label="Income Expense")




#Check how Expense is varying with Age
x = MyData["Age"]
y=  MyData["Expense"]


plt.scatter(x, y, label="Income Age")


# In[52]:


#check correltion matrix - to check the strength of variation bwtween two variables
correlation_matrix= MyData.corr().round(2)
f, ax = plt.subplots(figsize =(8, 4)) 
import seaborn as sns
sns.heatmap(data=correlation_matrix, annot=True)




################Step 4-feature engineering#######################
#Normalization/scaling of data - understanding scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(MyData)
scaled_data


#converting data back to pandas dataframe
MyData_scaled = pd.DataFrame(scaled_data)
MyData_scaled.columns = ["Age","Income","Expense"]




#Separating features and response
features = ["Income","Age"]
response = ["Expense"]
X=MyData_scaled[features]
y=MyData_scaled[response]




#Dividing data in test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Importing neccesary packages
from sklearn.linear_model import LinearRegression
from sklearn import metrics




#Fitting lineaar regression model
model = LinearRegression()
model.fit(X_train, y_train)




#Checking accuracy on test data
accuracy = model.score(X_test,y_test)
print(accuracy*100,'%')




model.predict(X_test) #predcited values on test data



#Dumping the model object
import pickle
pickle.dump(model, open('model.pkl','wb'))


#Reloading the model object
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[30000, 24]]))






