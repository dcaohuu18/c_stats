import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from data.read_data import read_last_time, read_n_recent
from helper_functions import graph_hist, estimate_pdf


MONTH_SEM_MAP = {mon: 'FA' if 8<=mon<=12 else 'SP' for mon in range(1, 13)} 
# assume there're only 2 semesters: Fall (FA) and Spring (SP)


def basics(course, current_term, cur_course_df):
    st.header('Basics')

    grades_df = cur_course_df.iloc[:, 8:-9]
    grades_df.dropna(how='all', axis=1, inplace=True) # drop unavailable grades

    hist_layout = graph_hist(grades_df, 'Distribution of grades', 'Grades', 'Number of students')
    st.bokeh_chart(hist_layout)

def advanced(course, current_term):
        st.header('Advanced')

def main():
    # Title 
    st.title('Welcome to C_STATS')

    # Select course:
    course_list = ['CS_320'] # hard coded # will need to read database
    course = st.selectbox('Select course:', course_list)

    today_year = datetime.today().year
    today_month = datetime.today().month
    current_term = '{}_{}'.format(MONTH_SEM_MAP[today_month], today_year)

    cur_course_df = read_n_recent(course, current_term)
    # st.dataframe(cur_course_df)

    # Functionality options:
    func_options = ['Basics', 'Advanced']
    func_choice = st.sidebar.radio('Options', func_options)

    if func_choice == 'Basics':
        basics(course, current_term, cur_course_df)

    elif func_choice == 'Advanced':
        advanced(course, current_term)

    # About:
    st.sidebar.header('About')
    st.sidebar.info('C_STATS is a learning management and analytics tool.')


if __name__ == '__main__':
    main()