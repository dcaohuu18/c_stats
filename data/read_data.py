import numpy as np 
import pandas as pd
import xlrd


def read_course(course_code, term_code): # return a df
    # If we change the database structure, just need to modify this 
    try:
        course_file_name = './data/{}.xlsx'.format(course_code)
        return pd.read_excel(course_file_name, sheet_name = term_code)

    except (FileNotFoundError, xlrd.biffh.XLRDError): # file or sheet (term) not found
        return pd.DataFrame() # return empty df

#####

def subtract_term(current, n_term):
    cur_sem, cur_yea = tuple(current.split('_')) # ex: FA_2019
    cur_yea = int(cur_yea)
    past_y = cur_yea - (n_term//2)

    # assume that each year only has 2 semesters: Spring (SP) & Fall (FA):
    if n_term%2 == 0:
        return '{}_{}'.format(cur_sem, past_y)      
    elif n_term%2 == 1:
        if cur_sem=='FA':
            return 'SP_{}'.format(past_y)            
        elif cur_sem == 'SP':
            return 'FA_{}'.format(past_y-1)

def term_range(current, n_term): # return a list of the past n_term starting at current (inclusive)
    terms_passed = 0
    term_list = []

    while terms_passed <= n_term:
        temp_term = subtract_term(current, terms_passed)
        term_list.append(temp_term)

        terms_passed += 1
    
    return term_list 

def read_last_time(course_code, current, cutoff=10): # read the last time this course is offered
    last_term = subtract_term(current, 1)
    terms_passed = 0

    while terms_passed <= cutoff:
        last_time_df = read_course(course_code, last_term)
        if not last_time_df.empty:
            break
        
        terms_passed += 1
        last_term = subtract_term(last_term, 1)

    return last_time_df

def read_n_recent(course_code, current_term, n_term=0):
    all_term_df = pd.DataFrame()
    for t in term_range(current_term, n_term):
        temp_df = read_course(course_code, t)
        all_term_df = pd.concat([all_term_df, temp_df])
    return all_term_df
    