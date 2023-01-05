import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

#style = {'description_width': 'initial'}

st.title("AutoML")

input_csv = st.file_uploader('', type=['csv'], accept_multiple_files=False)

def train_model(input_df):
    x = input_df.iloc[:, :-1]
    y = input_df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, penalty='l2', C=0.04, random_state=0),
    'navie_bayes': GaussianNB(),
    'svc': SVC(random_state=0),
    'Random_Forest': RandomForestClassifier(random_state=0),
    'ada_boost': AdaBoostClassifier(learning_rate=0.01, random_state=0),
    'gradient_boost': GradientBoostingClassifier(random_state=0),
    'sgd': SGDClassifier(random_state=0),
    'Bagging_Classifer': BaggingClassifier(random_state=0),
    'knn_classifier': KNeighborsClassifier()
    }
    accuracy = []
    model_name_1 = []
    def train_model(model, model_name, x=x_train, y=y_train, x_test = x_test):
        model = model.fit(x, y)
        y_pred = model.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        model_name_1.append(model_name)

    for model_name, model in models.items():
        train_model(model, model_name)

    my_dict = dict(zip(model_name_1, accuracy))
    my_dict

    max_model_name = my_dict['Logistic_Regression']
    best_model_name = ''
    for i in my_dict.keys():
        if max_model_name < my_dict[i]:
            max_model_name = my_dict[i]
            best_model_name = i

    model = models[best_model_name]
    st.download_button("Download Model", data=pickle.dumps(model), file_name="model.pkl")

if input_csv:
    input_df = pd.read_csv(input_csv)
    st.subheader('Uploded Dataset')
    st.write(input_df.head())
    if input_df.index[-1] > 5000:
        input_df = input_df.sample(n=5000)
        train_model(input_df)
    else:
        train_model(input_df)

    