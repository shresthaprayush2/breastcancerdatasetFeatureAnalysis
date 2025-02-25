# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Load the breast cancer dataset
cancer_data = load_breast_cancer()
# Creating a dataframe
df = pd.DataFrame(cancer_data.data,columns=cancer_data.feature_names)
# Adding a target set
df['target'] = cancer_data.target

# Creating a seed value to reproduce same results.
SEED = 10

# Creating a dictionary to structure the data clearly
completeDictionary = []


# A simple function that take accuracy, algorithm and data and adds it to a dictionary
def addIntoDictionary(accuracy,algorithm,data="ALL"):
    data = {
        'Algorithm':algorithm,
        'Accuracy':accuracy*100,
        'Data':data
    }
    completeDictionary.append(data)

# Simple and quick data analysis
print("Data Check\n")
print("Performing General Analysis")
print("Size of the dataframe ",df.size)
print("Shape of the dataframe ",df.shape)
print("Number of null values Across Columns \n",df.isna().sum())
print("Data Type Information \n",df.dtypes)

# A quick analysis of the target set
print("Checking the data type of the target set ",df['target'].dtypes)
print("Checking the unique values ",df['target'].unique())
print("Checking the count of values ",df['target'].value_counts())


# Creating a function to change 0 and 1 to malignant and bening.
# This function will be used in map
def changeText(x):
    if x == 0:
        return "malignant"
    else:
        return "benign"
# Using map to change all the row
df['targetUpdated'] = df['target'].map(changeText)
print(df['targetUpdated'].sample(10))

# A simple count plot to check the results
sns.countplot(df,x='targetUpdated')
plt.title("Histogram of Cancerous or not")
plt.xlabel("Count")
plt.ylabel("Class")
plt.show()

# Describing the data to get a quick statistical analysis
print(df.describe(include='all'))

# Separating the data into feature and target matrix
y = df['target']
X = df.drop(columns=['target','targetUpdated'])
print(X.head(10).to_string())

# Separating the data into training and validation set
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=80,random_state=10)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Using a LR and RF model with no hyperparameters to hide complexity
modelLR = LogisticRegression()
modelRF = RandomForestClassifier()

# Since this is a supervised learning model we pass X as well as Y
modelLR.fit(X_train,y_train)
# Predict the data
predY = modelLR.predict(X_test)
# Getting the accuracy
accuracy = accuracy_score(y_test,predY)
# Calling the function we made earlier
addIntoDictionary(accuracy,"LR")
# print(f'The Accuracy Score for the model is {accuracy*100}')

# Doing the same steps for random forest
modelRF.fit(X_train,y_train)
predY = modelRF.predict(X_test)
accuracy = accuracy_score(y_test,predY)
addIntoDictionary(accuracy,"RF")
# print(f'The Accuracy Score for the model is {accuracy*100}')


# Get feature importance (coefficients)
feature_importance = pd.Series(modelLR.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)

# Display top 10 features
print("Top 10 Most Influential Features in Predicting Breast Cancer:")
print(feature_importance.head(10))

# Get feature importance
rf_feature_importance = pd.Series(modelRF.feature_importances_, index=X.columns).sort_values(ascending=False)

# Display top 10 features
print("Top 10 Most Influential Features (Random Forest):")
print(rf_feature_importance.head(10))

# Creating an updated dataset with only the important feature
columnsLr = feature_importance.head(10).index.values

# For the RF
columnsrF = rf_feature_importance.head(10).index.values
print("This is this")
print(columnsrF)

# Getting the X and Y dataset
xLr =X[columnsLr]
print(xLr.head(10).to_string())

xRf = X[columnsrF]
print(xRf.head(10).to_string())

XtrainNewLr,XtestNewLr,yTrainNewLr,YtestNewLr = train_test_split(xLr,y,train_size=80,random_state=101)
print("After Selecting Only the best features")
# Using a LR model with no hyperparameters
# Since this is a supervised learning model we pass X as well as Y
modelLR.fit(XtrainNewLr,yTrainNewLr)
# Predict the data
predY = modelLR.predict(XtestNewLr)
# Getting the accuracy
accuracy = accuracy_score(YtestNewLr,predY)
addIntoDictionary(accuracy,"LR","TOP10")

# print(f'The Accuracy Score for the model is {accuracy*100}')


XtrainNewRf,XtestNewRf,yTrainNewRf,YtestNewRf = train_test_split(xRf,y,train_size=80,random_state=101)
modelRF.fit(XtrainNewRf,yTrainNewRf)
predY = modelRF.predict(XtestNewRf)
accuracy = accuracy_score(YtestNewRf,predY)
addIntoDictionary(accuracy,"RF","TOP10")

# print(f'The Accuracy Score for the model is {accuracy*100}')

# Creating a PCA model to extract new features from the data
from sklearn.decomposition import PCA
pcaModel = PCA(n_components=10)
X_trainPCA = pcaModel.fit_transform(X_train)
# Transform the test data using the already fitted PCA model
X_testPCA = pcaModel.transform(X_test)



# Fitting the new data in our test algorithms
modelLR.fit(X_trainPCA,y_train)
predY = modelLR.predict(X_testPCA)
accuracy = accuracy_score(y_test,predY)
addIntoDictionary(accuracy,"LR","PCA")
# print(f'The Accuracy Score for the model after PCA LR {accuracy*100}')


modelRF.fit(X_trainPCA,y_train)
predY = modelRF.predict(X_testPCA)
accuracy = accuracy_score(y_test,predY)
addIntoDictionary(accuracy,"RF","PCA")

# print(f'The Accuracy Score for the model after PCA {accuracy*100}')



# Creating the final dataframe
finalDataDictionary = pd.DataFrame(completeDictionary)
print("Final Results!!!!")

print(finalDataDictionary)


# Plotting the results
plt.figure(figsize=(8, 5))
sns.barplot(x="Data", y="Accuracy", hue="Algorithm", data=finalDataDictionary, palette="coolwarm")
plt.title("Accuracy of Different Algorithms on Features")
plt.ylabel("Accuracy (%)")
plt.xlabel("Feature Set")
# Make sure to change the limit accordingly to make sure to select the region where you can clearly see the changes
plt.ylim(80, 100)
plt.legend(title="Algorithm")
plt.show()


plt.figure(figsize=(8, 5))
sns.lineplot(x="Data", y="Accuracy", hue="Algorithm", marker="o", data=finalDataDictionary)
plt.title("Accuracy Trends Across Feature Sets")
plt.ylabel("Accuracy (%)")
plt.xlabel("Feature Set")
# Make sure to change the limit accordingly to make sure to select the region where you can clearly see the changes
plt.ylim(90, 100)
plt.grid()
plt.legend(title="Algorithm")
plt.show()



