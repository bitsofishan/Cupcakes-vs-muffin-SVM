import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(font_scale=1.2)

recipes=pd.read_csv(r'C:\Users\Ishan\Documents\Python Scripts\Datasets\muffin-cupcake-Scones\recipes_muffins_cupcakes.csv')
print(recipes.head())

sns.lmplot("Flour","Milk",data=recipes,hue='Type',
           palette='Set1',fit_reg=False,scatter_kws={"s":70});

type_label=np.where(recipes["Type"]=="Muffin",0,1)
recipes_features=recipes.columns.values[1:].tolist()
print(recipes_features)

ingredients=recipes[["Flour","Milk"]].values
print(ingredients)

model=svm.SVC(kernel='linear')
model.fit(ingredients,type_label)

w=model.coef_[0]
a=-w[0]/w[1]
xx=np.linspace(30,60)
yy=a*xx-(model.intercept_[0])/w[1]

b=model.support_vectors_[0]
yy_down=a*xx+(b[1]-a*b[0])
b=model.support_vectors_[-1]
yy_up=a*xx+(b[1]-a*b[0])

sns.lmplot("Flour","Milk",data=recipes,hue='Type',
           palette='Set1',fit_reg=False,scatter_kws={"s":70});
plt.plot(xx,yy,linewidth=2,color="black")
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')

def muffinvscupcke(flour,sugar):
    if(model.predict([[flour,sugar]]))==0:
        print("Muffin Recipe")
    else:
        print("Cupcake recipe")
muffinvscupcke(50,20)        
     