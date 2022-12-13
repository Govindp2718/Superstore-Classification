import streamlit as st
import numpy as np
import utils
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix




from sklearn.metrics import precision_score, recall_score

st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    st.title("Superstore Dataset Classification Project")
    #st.sidebar.title("Classification Web App")

    st.sidebar.header("Project for CSCI-657")

    st.sidebar.markdown(" ")
    st.sidebar.markdown("*This project is done as a part of the final project for the course Introduction to Data Mining*")

    st.sidebar.markdown("**Authors**: Govind Pande, Fernando Rivero Laseca, Kuldeep Malhotra & Jayuan Pendley")
    #st.sidebar.markdown("**NYIT ID**: 1302516")
    #st.markdown("Profit or Loss prediction using Machine Learning")
    st.sidebar.markdown("Choose from 4 classifiers")



    df = utils.load_data()
    x_train, x_test, y_train, y_test = utils.split(df)
    class_names = ["gain", "loss"]

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Naive Bayes","Decision Tree Classification","Random Forest Classification", "K Nearest Neighbors","Support Vector Machine (SVM)", "All Models Evaluation"))


    df_results=pd.DataFrame(columns=["Classifier_name","Accuracy","Precision","Sensitivity","Specificity"])

    def print_all_model_evalutions(df,model_name,cm):
        total1=sum(sum(cm))
    #####from confusion matrix calculate accuracy
        accuracy1=(cm[0,0]+cm[1,1])/total1
        st.write('Accuracy : ', accuracy1)
    
        precision=(cm[0,0]/(cm[0,0]+cm[1,0]))
        st.write('Precision is: ',precision)
        sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
        st.write('Sensitivity/Recall : ', sensitivity1 )

        specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
        st.write('Specificity : ', specificity1)
        df=df.append({"Classifier_name":model_name,"Accuracy":accuracy1,"Precision":precision,"Sensitivity":sensitivity1,"Specificity":specificity1},ignore_index=True)
        return df


    def print_all_model_evalutions1(df,model_name,cm):
        total1=sum(sum(cm))
    #####from confusion matrix calculate accuracy
        accuracy1=(cm[0,0]+cm[1,1])/total1
        #st.write('Accuracy : ', accuracy1)
    
        precision=(cm[0,0]/(cm[0,0]+cm[1,0]))
        #st.write('Precision is: ',precision)
        sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
        #st.write('Sensitivity/Recall : ', sensitivity1 )

        specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
        #st.write('Specificity : ', specificity1)
        df=df.append({"Classifier_name":model_name,"Accuracy":accuracy1,"Precision":precision,"Sensitivity":sensitivity1,"Specificity":specificity1},ignore_index=True)
        return df




    if classifier == 'Support Vector Machine (SVM)':
        #st.sidebar.subheader("Model Hyperparameters")
        #C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        #kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        #gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            st.markdown("Support Vector Machines are a machine learning algorithm used for classification. It primarily works by maximizing the margin of a hyperplane between points. SVM consists of hyperplane, soft margin, maximum margin and the kernel function. SVM are considered to be kernel based on a specific kernel function. A kernel function can be appropriately selected to fit either linear or nonlinear classifiers across linear or nonlinear decision boundaries.")
            model = SVC()
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            #st.write("Accuracy: ", accuracy.round(2))
            #st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            #st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            df_results=print_all_model_evalutions(df_results,"SVM",confusion_matrix(y_test,y_pred))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)



    if classifier == 'Naive Bayes':
        #st.sidebar.subheader("Model Hyperparameters")
        #C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='Lr')
        #max_iter = st.sidebar.slider("Maximum no. of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Naive Bayes")
            st.markdown("Naive Bayes classifiers are a family of probabilistic classifiers based on applying Bayes theorem with strong independence assumptions between the features. They are among the simplest Bayesian network models, but coupled with kernel density estimation, they can achieve high accuracy levels. It´s used in a wide variety of classification tasks.")
            model = GaussianNB()
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            #st.write("Accuracy: ", accuracy.round(2))
            #st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            #st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            df_results=print_all_model_evalutions(df_results,"Naive Bayes",confusion_matrix(y_test,y_pred))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)


    if classifier == 'Random Forest Classification':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("This is the number of trees in the forest", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=2, key='max_depth')
        #bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap')
        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve")

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            st.markdown("Random Forest is an ensemble tree method proposed in 2001 by L. Brieman, which seeks to divide a problem into smaller subsets and then aggregate them together by majority voting schema. As a general rule, Random forest grows randomized trees as it continues to split according to the CART algorithm on randomized subsets of a dataset. Once we set a prediction we use each randomized forest and take the rule by majority voting.")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            #st.write("Accuracy: ", accuracy.round(2))
            #st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            #st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            df_results=print_all_model_evalutions(df_results,"Random Forest",confusion_matrix(y_test, y_pred))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)
                                         
                                         
                                         
    if classifier == "Decision Tree Classification':
        st.sidebar.subheader("Model Hyperparameters")
        #n_estimators = st.sidebar.number_input("This is the number of trees in the forest", 100, 5000, step=10,key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100,5, step=2, key='max_depth')
        #bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap')
        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Decision Tree Results")
            #st.markdown("Random Forest is an ensemble tree method proposed in 2001 by L. Brieman, which seeks to divide a problem into smaller subsets and then aggregate them together by majority voting schema. As a general rule, Random forest grows randomized trees as it continues to split according to the CART algorithm on randomized subsets of a dataset. Once we set a prediction we use each randomized forest and take the rule by majority voting.")
            model = DecisionTreeClassifier(max_depth=max_depth, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            #st.write("Accuracy: ", accuracy.round(2))
            #st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            #st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            df_results=print_all_model_evalutions(df_results,"Decision Tree",confusion_matrix(y_test, y_pred))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == 'K Nearest Neighbors':
        st.sidebar.subheader("Model Hyperparameters Results")
        n_neighbors = st.sidebar.number_input("This is the number of neighbors", 3, 15, step=1,
                                               key='n_neighbors')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("K Nearest Neighbors")
            st.markdown("K Nearest Neighbor is a lazy learner instance-based classification algorithm. It classifies by selecting a value of k and uses different distance metrics such as Manhattan or Euclidean Distance to assign a value based on majority voting between that associated class that it is closest to. KNN remains a simple method, but has shortcomings when data cannot be easily assigned to one class as when there are even splits or outlying data.")
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            #st.write("Accuracy: ", accuracy.round(2))
            #st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            #st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            df_results=print_all_model_evalutions(df_results,"KNN",confusion_matrix(y_test,y_pred))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)
    #st.dataframe(df_results)





    if classifier == 'All Models Evaluation':
        st.title("Model Evaluation")
        st.markdown("All the models have a good prediction accuracy, exceeding 80% of prediction capacity. At this point, both Random Forest and Decision Tree are better, at 95%. Both models have great precision and recall, but we choose Decision Tree because of its ability to avoid false alarms.")



        model = SVC()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)        

        
        df_results=print_all_model_evalutions1(df_results,"SVM",confusion_matrix(y_test,y_pred))
        

        model = GaussianNB()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)        
        df_results=print_all_model_evalutions1(df_results,"Naive Bayes",confusion_matrix(y_test,y_pred))
        

        model = RandomForestClassifier(n_estimators=5, max_depth=7, n_jobs=-1)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        df_results=print_all_model_evalutions1(df_results,"Random Forest",confusion_matrix(y_test, y_pred))
        

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        df_results=print_all_model_evalutions1(df_results,"KNN",confusion_matrix(y_test,y_pred))
        


        st.dataframe(df_results)






    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Superstore Data Set (Classification)")
        st.write(df)


if __name__ == '__main__':
    main()
