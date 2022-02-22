import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Osman\Desktop\VBM683MakineÖğrenmesi\earthquake.csv")
print(data)
print("---")

data['date'] =  pd.to_datetime(data['date'],
                              format='%Y.%m.%d')

datatest = pd.read_csv(r"C:\Users\Osman\Desktop\VBM683MakineÖğrenmesi\earthquake-test.csv")
print(datatest)
print("---")


X = data.iloc[:,:-1]
print(X)
print("---")

Xtest = datatest.iloc[:,:-1]
print(Xtest)
print("---")

y = data.iloc[:,-1:]
print(y)
print("---")

ytest = datatest.iloc[:,-1:]
print(ytest)
print("---")

X = data.iloc[:,:-1].values
print(X)
print("---")

Xtest = datatest.iloc[:,:-1].values
print(Xtest)
print("---")

y = data.iloc[:,-1:].values
print(y)
print("---")

ytest = datatest.iloc[:,-1:].values
print(ytest)
print("---")

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, 
                                                    test_size=0.2)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train[:, 1:] = ss.fit_transform(X_train[:, 1:])
X_validation[:, 1:] = ss.transform(X_validation[:, 1:])
Xtest[:, 1:] = ss.transform(Xtest[:, 1:])


# raise ValueError("Unknown label type: %r" % y_type)
# ValueError: Unknown label type: 'continuous' 
# Hatası aldığım için aşağıdaki şekile continous'u int'e çevirdim.

y_train = y_train.astype('int')
y_validation = y_validation.astype('int')


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train[:, 1:], y_train)

y_pred = classifier.predict(X_validation[:, 1:])
y_predfuture = classifier.predict(Xtest[:, 1:])

print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_validation, y_pred))
print(classification_report(y_validation, y_pred))

error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train[:, 1:], y_train)
    pred_i = knn.predict(X_validation[:, 1:])
    error.append(np.mean(pred_i != y_validation))
    
plt.figure(figsize=(30, 30))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
          markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

dataframe = pd.DataFrame(y_predfuture)
lat = pd.read_csv(r"C:\Users\Osman\Desktop\VBM683MakineÖğrenmesi\earthquake-test.csv")["lat"]
long = pd.read_csv(r"C:\Users\Osman\Desktop\VBM683MakineÖğrenmesi\earthquake-test.csv")["long"]
dataframe = dataframe.join(lat)
dataframe = dataframe.join(long)
print(dataframe)

dataframe.to_excel("output.xlsx")  

                                                 







