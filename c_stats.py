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
    st.subheader('Summary figures')
    summary_table(cur_course_df, grade_df)
    st.text('')

    graph_choice = st.selectbox('Select variable for visualization:', ['Grades', 'Attendance', 'Submission', 
                                                                        'LMS Time', 'LMS Messages'])
    st.subheader(graph_choice)

    # Grades:
    if graph_choice=='Grades':
        st.info('Click on the legend to select the desired assessment')
        # switch to long form:
        grade_df = grade_df.melt(var_name='assessment', value_name='grade')
        graph_multi_hist(grade_df, x_field='grade', color_field='assessment', max_nbins=20, x_title='Grade')

    # Attendance:
    elif graph_choice=='Attendance':
        st.text('')
        graph_hist(cur_course_df, 'attendance', x_title='Attendance percentage')

    # Submission Status:
    elif graph_choice=='Submission':
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
        graph_submission(submission_df)

    # LMS Time:
    elif graph_choice=='LMS Time':
        st.text('')
        LMS_time_df = cur_course_df[['name', 'LMS_time', 'LMS_accesses']].copy()
        # switch to long form:
        LMS_time_df = LMS_time_df.melt('name', var_name='type', value_name='amount')
        LMS_time_df['type'] = LMS_time_df.type.apply(lambda x: 'weekly accesses' if x=='LMS_accesses' else 'session time (mins)')
        graph_LMS_time(LMS_time_df)

    # LMS Messages:
    elif graph_choice=='LMS Messages':
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
from helper_functions import knn_modeller, knn_predictor, ols_modeller, ols_predictor
from helper_functions import normalize, reweigh, modify_cur_course
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder
import st_state_patch

# map from the prediction preference to the method used:
PREF_METHOD_MAP = {
    'Regression of percentage grades': (ols_modeller, ols_predictor),
    'Classification of letter grades': (knn_modeller, knn_predictor)}
 

def advanced(course, current_term):
    st.header('Advanced')

    st.subheader('Preprocessing')
    # select the number of terms to retrieve:
    n_past_term = st.number_input("How many terms to retrieve?", min_value=1, max_value=10, value=5)
    course_df = read_n_recent(course, current_term, n_term=n_past_term)
    st.text('')

    # calculate age at the time of the course:
    course_year = course_df.term_code.apply(lambda x: re.findall("\d+", x)[0]).astype(int)
    course_df['age'] = course_year - course_df.birth_dt.dt.year

    # encode gender and race:
    le = LabelEncoder()
    course_df['gender'] = le.fit_transform(course_df['gender'])
    course_df['race'] = le.fit_transform(course_df['race'])

    # separate current course from previous courses:
    cur_course_df = course_df.loc[course_df.final_grade.isna()]
    pre_courses_df = course_df.loc[~course_df.final_grade.isna()]

    # normalize and reweigh: 
    norm_reweigh = st.checkbox('Normalize and reweigh (recommended)', value=True) 
    if norm_reweigh:
        assessments = st.multiselect('What will be counted towards the final grade?',
                                    options=['exam', 'home', 'quiz', 'attendance', 'LMS_mess', 'LMS_accesses', 'LMS_time'],
                                    default=['exam', 'home', 'quiz', 'attendance'])
        weights_dict = {} 
        for a in assessments:
            weights_dict[a] = st.number_input('Weight of {} (%):'.format(a), min_value=1, max_value=100, value=50)

        normalize(cur_course_df, weights_dict)   
        normalize(pre_courses_df, weights_dict)
        reweigh(pre_courses_df, weights_dict)

        st.markdown('<hr>', unsafe_allow_html=True) # horinzontal rule

    # drop unavailable and unrelevant vars:
    rel_vars = cur_course_df.dropna(how='all', axis=1)
    rel_vars.drop(columns=['id', 'f_name', 'l_name', 'birth_dt', 'address', 'email', 'course_code', 'term_code'], inplace=True)
    
    if norm_reweigh:
        drop_cols = []
        for ass in weights_dict.keys():
            r = re.compile("^{}\d+$".format(ass))
            drop_cols += list(filter(r.match, rel_vars.columns))
        # we drop columns like 'home1', 'home2' since we only consider 'home' (normalized)

        rel_vars.drop(columns=drop_cols, inplace=True)
        selected_vars = st.multiselect('Select variables to consider:', options=list(rel_vars.columns), 
                                        default=list(rel_vars.columns)[2:])
    else:
        selected_vars = st.multiselect('Select variables to consider:', options=list(rel_vars.columns), 
                                        default=list(rel_vars.columns)[2:-1])

    if not selected_vars:
        st.error("Please select at least one variable!")
    
    st.subheader('Modelling')
    # Create a session state to preserve the changes across sessions
    session_state = st.State() 
    if not session_state:
        session_state.new_cur_course_df = cur_course_df.copy()
        session_state.change_by_slider_id = 0
    
    # Adjust current course's variables:
    adjustable = set(selected_vars)
    adjustable.difference_update(['age', 'gender', 'race']) # remove age, gender, and race
    adjust_var = st.selectbox('Select variable to adjust:', list(adjustable))
    
    change_by_slider = st.empty() # placeholder
    apply_to = st.number_input('Apply this change to the lowest (%):', min_value=0, max_value=100, value=25, step=5)
    
    if st.button("Reset all changes"):
        session_state.new_cur_course_df = cur_course_df.copy()
        session_state.change_by_slider_id += 1

    change_by = change_by_slider.slider('Change {} by (%):'.format(adjust_var), min_value=-100, max_value=100, 
                                        value=0, key=session_state.change_by_slider_id)  
    modify_cur_course(cur_course_df, session_state.new_cur_course_df, adjust_var, change_by, apply_to)

    # Select prediction preference and run model:
    prediction_pref = st.selectbox('Select your prediction preference:', list(PREF_METHOD_MAP.keys()))
    modeller_used, predictor_used = PREF_METHOD_MAP[prediction_pref]

    model, metrics = modeller_used(pre_courses_df, selected_vars) 
    ## for KNN, the metric is the accuracy
    ## for OLS, the metrics are the correlation df and the scatter plot  
    if st.button('Run model'):
        predictor_used(session_state.new_cur_course_df, selected_vars, model, *metrics)


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