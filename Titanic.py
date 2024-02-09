import streamlit as s
import pandas as p
from sklearn import linear_model
import numpy as n
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import accuracy_score as as1
from sklearn.preprocessing import StandardScaler

##     https://discuss.streamlit.io/t/no-module-named-sklearn/9218/20

##     https://titanicsurvival-3crbwcweyojzmicx43oaqf.streamlit.app/
def model():
    a=p.read_csv('titanic_data.csv')
    gend= {"male": 0, "female": 1}
    ports = {"S": 0, "C": 1, "Q": 2}
    
    
    d=a[['Fare','Age','Survived','Parch','SibSp','Embarked','Pclass','Sex']].dropna()
    d['Sex']=d['Sex'].map(gend) 
    d['Embarked']=d['Embarked'].map(ports)
    
    x,y=n.array(d[['Age','Fare','Parch','SibSp','Embarked','Pclass','Sex']]).reshape([-1,7]),n.array(d['Survived'])
    xn,xt,yn,yt=tts(x,y,test_size=0.3, random_state=42) 
    knc=knc(5)
    mod=knc.fit(xn,yn) 
    return mod


if __name__ == "__main__": 
    s.set_page_config(page_title='Titanic Survival',layout='wide') 
    s.title("Titanic Survival Prediction") 
    
    cl=s.columns([1,3,1]) 
    with cl[0]:
        gen=s.selectbox("Enter Gender",['Male','Female']) 
    with cl[1]:
        por=s.selectbox("Please Enter the port Embarked",['Cherbourg',  'Queenstown', 'Southampton']) 
    with cl[2]:
        cls=s.selectbox("Please Enter the Class of Passenger",[1,2,3]) 
    cl=s.columns([1,1])
    with cl[0]:
        parch=s.number_input("Enter number of parents / children boarded",min_value=0, max_value=9, step=1)
    with cl[1]:
        sib=s.number_input("Enter number of siblings / spouses ",min_value=0, max_value=9, step=1) 
    age=s.slider("Enter the age of Passenger",min_value=0, max_value=90)
    fare=s.slider("Enter the Fare charges of Passenger",min_value=0, max_value=750)
