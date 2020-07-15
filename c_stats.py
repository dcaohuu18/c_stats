import streamlit as st
import numpy as np
import pandas as pd
from data.read_data import read_last_time, read_n_recent


#=======================================================================
from helper_functions import estimate_pdf, summary_table, make_hist, graph_multi_hist, graph_hist
from helper_functions import graph_donut, graph_submission, graph_LMS_time


def basics(course, current_term, cur_course_df):
    st.header('Basics')

    # combine f_name and l_name into name:
    cur_course_df['name'] = cur_course_df['f_name'] + ' ' + cur_course_df['l_name']

    grade_df = cur_course_df.iloc[:, 8:-11]
    grade_df.dropna(how='all', axis=1, inplace=True) # drop unavailable grades

    # Summary:
    st.subheader('Summary')
    summary_table(cur_course_df, grade_df)
    st.text('')

    # Grades:
    st.subheader('Grades')
    st.info('Click on the legend to select the desired assessment')
    # switch to long form:
    grade_df = grade_df.melt(var_name='assessment', value_name='grade')
    graph_multi_hist(grade_df, x_field='grade', color_field='assessment', max_nbins=20, x_title='Grade')

    # Attendance:
    st.subheader('Attendance')
    st.text('')
    graph_hist(cur_course_df, 'attendance', x_title='Attendance percentage')

    # Submission Status:
    st.subheader('Submission')
    submission_percentage = [['ontime', cur_course_df.ontime.mean()], 
                             ['late', cur_course_df.late.mean()], 
                             ['missed', cur_course_df.missed.mean()]]   
    submission_percentage = pd.DataFrame(submission_percentage, columns = ['status', 'percentage'])
    # Whole class: 
    graph_donut(submission_percentage, title='Submission status', value_fname='status')
    # Each student:
    submission_df = cur_course_df[['name', 'ontime', 'late', 'missed']].copy()
    # switch to long form:
    submission_df = submission_df.melt('name', var_name='status', value_name='percentage') 
    st.text('')
    st.text('')
    graph_submission(submission_df)

    # LMS Time:
    st.subheader('LMS Time')
    st.text('')
    LMS_time_df = cur_course_df[['name', 'LMS_time', 'LMS_accesses']].copy()
    # switch to long form:
    LMS_time_df = LMS_time_df.melt('name', var_name='type', value_name='amount')
    LMS_time_df['type'] = LMS_time_df.type.apply(lambda x: 'weekly accesses' if x=='LMS_accesses' else 'session time (mins)')
    graph_LMS_time(LMS_time_df)

    # LMS Messages:
    st.subheader('LMS Messages')
    # make histogram of LMS_mess:
    LMS_mess_df = cur_course_df[['name', 'LMS_mess']].copy()
    hist_df = make_hist(LMS_mess_df.LMS_mess, bins=[0, 2, 5, 8, np.inf])
    hist_df.iat[-1, -1] = '≥ 8'
    hist_df.rename(columns={'interval': 'messages'}, inplace=True) # rename to plot
    # add percentage and graph:
    hist_df['percentage'] = hist_df['count']/hist_df['count'].sum() * 100
    mean_mess = LMS_mess_df.LMS_mess.mean()
    graph_donut(hist_df, title='Messages distribution', 
                center_text='{:.1f} per student'.format(mean_mess), value_fname='messages')


#==============================================================================
from helper_functions import knn, ols_reg, multino_reg
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder


# map from the method's name (string) to its implementation (function):
METHOD_MAP = {
    'K nearest neighbors': knn,
    'Multinomial regression': multino_reg,
    'Ordinary least squares regression': ols_reg
} 

def advanced(course, current_term):
    st.header('Advanced')

    n_past_term = st.number_input("How many terms to retrieve?", min_value=1, max_value=10, value=5)
    course_df = read_n_recent(course, current_term, n_term=n_past_term)

    # calculate age at the time of the course:
    course_year = course_df.term_code.apply(lambda x: re.findall("\d+", x)[0]).astype(int)
    course_df['age'] = course_year - course_df.birth_dt.dt.year

    # encode gender and race:
    le = LabelEncoder()
    course_df['gender'] = le.fit_transform(course_df['gender'])
    course_df['race'] = le.fit_transform(course_df['race']) 

    cur_course_df = course_df.loc[course_df.final_grade.isna()]
    pre_courses_df = course_df.loc[~course_df.final_grade.isna()]

    # drop unavailable and unrelevant vars:
    rel_vars = cur_course_df.dropna(how='all', axis=1)
    rel_vars.drop(columns=['id', 'f_name', 'l_name', 'birth_dt', 'address', 'email', 'course_code', 'term_code'], inplace=True)
    selected_vars = st.multiselect('Select variables to consider:',  options=list(rel_vars.columns), 
                                    default=list(rel_vars.columns)[2:-1])

    # st.subheader('Statistical analysis')
    
    st.subheader('Modelling')
    st.info(
    '''
    K nearest neighbors and Multinomial regression is for classification of letter grades. 
    Ordinary least square regression is for regression of percentage grades.
    ''')
    selected_method = st.selectbox('Select method:', list(METHOD_MAP.keys()))
    selected_method = METHOD_MAP[selected_method]
    selected_method(cur_course_df, pre_courses_df, selected_vars)


#==============================================================================


MONTH_SEM_MAP = {mon: 'FA' if 8<=mon<=12 else 'SP' for mon in range(1, 13)} 
# assume there're only 2 semesters: Fall (FA) and Spring (SP)

def main():
    # Title 
    st.title('Welcome to C_STATS 👋')

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

    # About:
    st.sidebar.header('About')
    st.sidebar.info('C_STATS is a learning management and analytics tool.')

    if func_choice == 'Basics':
        basics(course, current_term, cur_course_df)

    elif func_choice == 'Advanced':
        advanced(course, current_term)


if __name__ == '__main__':
    main()