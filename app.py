import pandas as pd
import numpy as np
import streamlit as st
import time
import pickle
from streamlit_option_menu import option_menu

import scipy.stats as stats
from scipy.stats.mstats import winsorize
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title='Graduate Admission',page_icon='🏫',layout="wide")
selected = option_menu(
    menu_title = "Graduate Student Admission",
    options = ["Home", "Dataset","Dashboard","Prediction", "Contact"],
    icons=["house", "hdd-stack-fill","graph-up","mortarboard-fill","info-circle-fill"],
    menu_icon = "cast",
    default_index= 0,
    orientation="horizontal",
    styles = {
        "container": {"padding": "0!important", "background-color": "#a4f4f9"},
        "icon": {"color": "White", "font-size": "30px"},
        "nav-link": {"font-size": "30px", "text-align": "center", "margin": "0px", "--hover-color": "#E0FFFF"},
        "nav-link-selected": {"background-color": "#FA8072"}
    }
)

@st.cache
def load_data():
    df = pd.read_csv('Admission_Predict_Ver1.1.csv')
    df.columns = df.columns.str.strip()
    df.set_index('Serial No.', inplace=True)
    df['LOR'] = winsorize(df['LOR'], limits=(0.005, 0))
    df['Chance of Admit'] = winsorize(df['Chance of Admit'], limits=(0.005, 0))
    return df
df = load_data()

# df['Chance of Admit'].plot(kind='hist', color='salmon');
# # --------------------------------------------------

if selected == "Home":
    st.markdown("# Graduate Student Admission")
    st.write("    ")
    st.markdown("### MOTIVATION")
    st.write("    ")

    st.write(" **Postgraduate degrees are becoming more and more desired degrees all over the world."
             " It is an advantage for the students to have an idea ahead about their probability of being admitted to a university, "
             "as a result, the students can work on enhancing the language test or the degree for their currently running courses and so on. "
             "In our project, we use a regression task to predict the student admission percentage.**")
    st.write(" ")

    st.markdown("### PROJECT OBJECTIVE")
    st.write("**My university acceptance calculator can help you to find the probability of getting accepted into a particular university based on your profile, "
             "and it is completely free. Enter your language scores and CGPA to see the predicted output."
             " This output will give you a fair idea about your chance of being admitted to a particular university.**")

if selected == "Dataset":
    st.markdown("# Dataset Details")
    st.markdown("### Explanation of the different features and the output of the dataset")
    st.write("    ")
    st.write(" **This dataset was built with the purpose of helping students in shortlisting universities with their profiles. "
             "The predicted output gives them a fair idea about their chances for a particular university. We use the dataset "
             "which is available in link below:**")
    st.write(":point_right: [Dataset_link](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)")
    st.write(" -------------------------------------------------------------------------------------------------------      ")

    st.markdown("### ATTRIBUTES OF DATASET")
    st.write("o	GRE Score (0 to 340)")
    st.write("o	TOEFL Score (0 to 120)")
    st.write("o	SOP ( Statement of Purpose) Strength(out of 5)")
    st.write("o	LOR(Letter of Recommendation) Strength(out of 5)")
    st.write("o	Research Experience ( 0 for no experience and 1 for having an experience)")
    st.write("o	Undergraduate CGPA is the average of grade points obtained in all the subject (out of 10)")
    st.write("o	Chance of Admit (range from 0 to 1) --> dependent variable")
    st.write(" ")
    st.write("The size of the dataset is 500 records and 8 columns and it contains several parameters which are considered "
             "important during the application for Masters Programs.")

    st.write("The table below shows a sample from our dataset :")
    st.table(df.head())
    st.write(" ")
if selected == "Dashboard":
    # st.set_page_config(page_title='Graduate Admission', page_icon='🏫')
    st.markdown("# Data Visualization")
    st.markdown("### This dashboard show the relations between the features in the dataset")
    st.write("    ")
    st.write("    ")

    st.write( " -----------------------------------------------------------------------------------------------------------  ")
    # -----------------------------------------------------------------------------------
    st.markdown("##### Chance of Admit based on CGPA and GRE Score as discriminator")
    fig = plt.figure(figsize=(22, 6))
    plt.title('Chance of Admit based on CGPA and GRE Score as discriminator')
    sns.scatterplot(df['CGPA'], df['Chance of Admit'], hue=df['GRE Score'], s=200)
    plt.show()
    st.pyplot(fig)
    # -------------------------------------------------------------------------------------
    st.write( " -----------------------------------------------------------------------------------------------------------  ")
    st.markdown("##### Chance of Admit based on CGPA and Research as discriminator")
    fig = plt.figure(figsize=(22, 6))
    plt.title('Chance of Admit based on CGPA and Research as discriminator')
    sns.scatterplot(df['CGPA'], df['Chance of Admit'], hue=df['Research'], s=200);
    st.pyplot(fig)
    # -------------------------------------------------------------------------------------
    st.write( " -----------------------------------------------------------------------------------------------------------  ")
    st.markdown("##### Chance of Admit based on CGPA and University Rating as discriminator")
    fig = plt.figure(figsize=(22, 6))
    plt.title('Chance of Admit based on CGPA and University Rating as discriminator')
    sns.scatterplot(df['CGPA'], df['Chance of Admit'], hue=df['University Rating'], s=200, palette="Set2");
    st.pyplot(fig)
    # -------------------------------------------------------------------------------------
    st.write( " -----------------------------------------------------------------------------------------------------------  ")
    st.markdown("##### Chance of Admit based on CGPA and SOP as discriminator")
    fig = plt.figure(figsize=(22, 6))
    plt.title('Chance of Admit based on CGPA and SOP as discriminator')
    sns.scatterplot(df['CGPA'], df['Chance of Admit'], hue=df['SOP'], s=200, palette="bright");
    st.pyplot(fig)
    # -------------------------------------------------------------------------------------
    st.write( " -----------------------------------------------------------------------------------------------------------  ")
    st.markdown("##### Chance of Admit based on CGPA and LOR as discriminator")
    fig = plt.figure(figsize=(22, 6))
    plt.title('Chance of Admit based on CGPA and LOR as discriminator')
    sns.scatterplot(df['CGPA'], df['Chance of Admit'], hue=df['LOR'], s=200, palette="bright");
    st.pyplot(fig)
    # -------------------------------------------------------------------------------------
    st.write( " -----------------------------------------------------------------------------------------------------------  ")
    st.markdown("##### Correlation Plot")
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, linewidths=0.05, fmt='.2f', cmap="magma")
    plt.show()
    st.pyplot(fig)
    # -------------------------------------------------------------------------------------
    st.write( " -----------------------------------------------------------------------------------------------------------  ")

    # -------------------------------------------------------------------------------------
    st.markdown("##### Percentage of candidates with Research Experience and 'Chance of Admit'>75%")
    top_OneFourth_df = df[df['Chance of Admit'] >= 0.75]
    research_df = top_OneFourth_df.groupby('Research').count()
    fig = plt.figure(figsize=(12, 10))
    plt.title("Percentage of candidates with Research Experience and 'Chance of Admit'>75%")
    colors = ['mistyrose', 'turquoise']
    explode = (0, 0.05)
    plt.pie(research_df['Chance of Admit'], labels=['Without Research Experience', 'With Research Experience'],
            autopct='%1.1f%%', startangle=90, colors=colors, explode=explode);
    st.pyplot(fig)
    # -------------------------------------------------------------------------------------
    st.write(" -----------------------------------------------------------------------------------------------------------  ")
    st.markdown("##### Chance of Admit vs University Rating")
    fig = plt.figure(figsize=(12, 10))
    sns.pointplot(df['University Rating'], df['Chance of Admit'])
    plt.title('Chance of Admit vs University Rating')
    plt.show()
    st.pyplot(fig)
    # -------------------------------------------------------------------------------------
    st.write( " -----------------------------------------------------------------------------------------------------------  ")
    st.markdown("##### University Rating (percentage wise)")
    colors = ['cyan', 'salmon', 'gold', 'orchid', 'orange']
    explode = [0.01, 0.01, 0.01, 0.01, 0.01]
    fig = plt.figure(figsize=(7, 7))
    plt.pie(df['University Rating'].value_counts().values, explode=explode,
            labels=df['University Rating'].value_counts().index, colors=colors, autopct='%1.1f%%')
    plt.title('University Rating', color='red', fontsize=25)
    plt.show()
    st.pyplot(fig)

    st.write( " -----------------------------------------------------------------------------------------------------------  ")
    st.write(" ")




if selected == "Prediction":
    # st.title("Graduate Student Admission")

    st.set_option('deprecation.showfileUploaderEncoding', False)
    model = pickle.load(open('regressor.pkl', 'rb'))


    def main():
        st.markdown(
            "<h1 style='text-align: center;  color: Black;background-color:#E0FFFF'>Student Admission Prediction</h1>",
            unsafe_allow_html=True)

#         st.markdown(
#             "<h3 style='text-align: center; color: salmon ;'>Drop in The required Inputs and we will do  the rest.</h3>",
#             unsafe_allow_html=True)
#         st.markdown("<h4 style='text-align: center; color: White;'>This Project is by Jaisingh Chauhan</h4>",
#                     unsafe_allow_html=True)
        st.write("-------------------------------------------------------------------------------------------------------  ")

        cgpa = st.slider("Input Your CGPA", 0.0, 10.0, 9.0,0.01)
        gre = st.slider("Input your GRE Score", 0, 340, 300,1)
        toefl = st.slider("Input your TOEFL Score", 0, 120, 100,1)
        research = st.slider("Do You have Research Experience (0 = NO, 1 = YES)", 0, 1, 1)
        uni_rating = st.slider("Rating of the University you wish to get in on a Scale 1-5", 1, 5, 4,1)
        SOP = st.slider("Rating of the SOP you Have", 1.0, 5.0, 4.0,0.5)
        LOR = st.slider("Rating of the LOR you Have", 1.0, 5.0, 4.0,0.5)

        # inputs = [[gre, toefl, uni_rating, SOP, LOR, cgpa, research]]

        if st.button('Check My Chances of Admission'):
            # # result = model.predict(inputs)
            # result = round(model.predict(inputs[0]*100, 3))
            # # updated_res = result.flatten().astype(float)
            st.success('Your chances of getting admission in postgraduate degree is {}%'.format(round(model.predict([[gre, toefl, uni_rating, SOP, LOR, cgpa, research]])[0]*100, 3)))

    if __name__ == '__main__':
        main()




if selected == "Contact":
    st.title("Made by Jaisingh Chauhan")
    st.title("Contact Details")
    st.write("    ")
    st.write("    ")

    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRLisL3IUg3KKSYgIMikl5CTDhOKV8Hg5nljAvjwgHXtsYuB8r660fH2cQs0O_yqy4cqCc&usqp=CAU.png",width=50)
    st.write(":point_right: [GitHub](https://github.com/Jaisingh-Chauhan)")
    st.write("    ")
    st.image("https://cdn3.iconfinder.com/data/icons/inficons/512/linkedin.png", width=50)
    st.write(":point_right: [Linkedin](https://www.linkedin.com/in/jaisingh-chauhan)")

