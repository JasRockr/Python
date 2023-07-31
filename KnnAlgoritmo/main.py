import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier


datos = pd.read_csv('./person_data.csv', header=0)
print(list())
genero = list(datos['Gender'])
altura = list(datos['Height'])
peso = list(datos['Weight'])

data = {'Masa': peso,
        'Altura': altura,
        'Genero': genero}
punto_nuevo = {'Masa': [70],
               'Altura': [1.82]}
df = pd.DataFrame(data)
punto_nuevo = pd.DataFrame(punto_nuevo)
# sns.scatterplot(df['Masa'], df['Altura'], hue=df['sex'])
ax = plt.axes()
ax.scatter(df.loc[df['Genero'] == 'Male', 'Masa'],
           df.loc[df['Genero'] == 'Male', 'Altura'],
           c="red",
           label="Hombre")
ax.scatter(df.loc[df['Genero'] == 'Female', 'Masa'],
           df.loc[df['Genero'] == 'Female', 'Altura'],
           c="blue",
           label="Mujer")
ax.scatter(punto_nuevo['Masa'],
           punto_nuevo['Altura'],
           c="black")
plt.xlabel("Masa")
plt.ylabel("Altura")
ax.legend()
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
X = df[['Masa', 'Altura']]
y = df[['Genero']]
knn.fit(X, y)
prediccion = knn.predict(punto_nuevo)
print(prediccion)