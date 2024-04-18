
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

model = GaussianNB(priors=None, var_smoothing=1e-9)
model.fit(X_train, y_train)

#Fazer a predição

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score: .2f}' )

df = pd.DataFrame({'Atual': y_test, 'Previsto': y_pred})

print(df)
