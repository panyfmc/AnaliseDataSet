
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data=[]
with open ('data.csv') as f:
    
    for line in f:
        data.append(line)

#print(data)
data2= pd.read_csv ('data.csv')       
print(data2)

#Separar o alvo 
data3 =  data2['Temperature']
print(data3.head())
data4 = data2.drop('Temperature', axis=1)

# Dividindo os dados em features (X) e target (y)
X = data4 # Suas features
y = data3 # Seus targets

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# test_size: especifica a proporção dos dados que serão utilizados como dados de teste (ex: 0.2 para 20%)
# random_state: um valor inteiro opcional que permite reproduzir a divisão exata toda vez que o código é executado

#Treinar o modelo
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# modelo Gaussian
model = GaussianNB(priors=None, var_smoothing=1e-9)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f'\nAccuracy GaussianNB: {score: .2f}\n')

df = pd.DataFrame({'Atual': y_test, 'Previsto': y_pred})  # dataframe para visualizar as predições

print(df)


# modelo KNeighborsClassifier 
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)

y_pred_knn = model_knn.predict(X_test)
score_knn = accuracy_score(y_test, y_pred_knn)
print(f'\nAccuracy KNeighborsClassifier: {score_knn:.2f}\n')

df_knn = pd.DataFrame({'Atual': y_test, 'Previsto': y_pred_knn})  # dataframe para visualizar as predições

print(df_knn)


# modelo DecisionTreeClassifier 
model_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
model_tree.fit(X_train, y_train)

y_pred_tree = model_tree.predict(X_test)
score_tree = accuracy_score(y_test, y_pred_tree)
print(f'\n Accuracy DecisionTreeClassifier: {score_tree:.2f}\n')

df_tree = pd.DataFrame({'Atual': y_test, 'Previsto': y_pred_tree})  # dataframe para visualizar as predições

print(df_tree)


# modelo LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# modelo LogisticRegression com os dados escalados
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled, y_train)

y_pred_lr = model_lr.predict(X_test_scaled)
score_lr = accuracy_score(y_test, y_pred_lr)
print(f'\nAccuracy LogisticRegression: {score_lr:.2f}\n')

df_lr = pd.DataFrame({'Atual': y_test, 'Previsto': y_pred_lr})

print(df_lr)


#print("Formato de X_train:", X_train.shape)
#print("Formato de X_test:", X_test.shape)
#print("Formato de y_train:", y_train.shape)
#print("Formato de y_test:", y_test.shape)
#print("Número de linhas nos dados originais:", data2.shape[0])

