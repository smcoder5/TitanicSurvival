import streamlit as s



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