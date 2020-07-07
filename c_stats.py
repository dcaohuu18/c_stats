import streamlit as st
from datetime import datetime
from data.read_data import read_last_time, read_n_recent


MONTH_SEM_MAP = {mon: 'FA' if 8<=mon<=12 else 'SP' for mon in range(1, 13)} 
# assume there're only 2 semesters: Fall (FA) and Spring (SP)


def basics(course, current_term):
	st.header('Basics')

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
	    basics(course, current_term)

    elif func_choice == 'Advanced':
        advanced(course, current_term)


if __name__ == '__main__':
    main()