import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats


def estimate_pdf(sample, n_points=101):
    lo = min(sample)
    hi = max(sample)
    kde = stats.gaussian_kde(sample) # using KDE algorithm

    xs = np.linspace(lo, hi, n_points)
    ds = kde.evaluate(xs) # density 

    return xs, ds


#=======================================================================
# BASICS - GRAPHS:

def summary_table(course_df, grade_df):
    # Mean statistics:
    mean_atten = course_df.attendance.mean()
    mean_grade = grade_df.mean().mean() # course_df.final_grade.mean()
    mean_ontime = course_df.ontime.mean()
    
    st.markdown(
        '''
        <table style=\"width:100%; text-align:center\">
            <tr style=\"font-size: 12px; color: gray; background-color:#D1ECF1\">
                <td style=\"border:2px solid white; border-radius:7px 7px 0 0\">Average attendance</td>
                <td style=\"border:2px solid white; border-radius:7px 7px 0 0\">Average grade</td>
                <td style=\"border:2px solid white; border-radius:7px 7px 0 0\">Average ontime</td>
            </tr>
            <tr style=\"font-size: 25px; color: gray; background-color:#D1ECF1\">
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius:0 0 7px 7px; width:34%\">{:.2f}%</td>
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius:0 0 7px 7px; width:32%\">{:.2f}</td>
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius:0 0 7px 7px; width:34%\">{:.2f}%</td>
            </tr>
        </table>
        '''.format(mean_atten, mean_grade, mean_ontime),
        unsafe_allow_html=True)

    # Median statistics:
    median_atten = course_df.attendance.median()
    median_grade = grade_df.median().median() # course_df.final_grade.median()
    median_ontime = course_df.ontime.median()

    st.markdown(
        '''
        <table style=\"width:100%; text-align:center\">
            <tr style=\"font-size: 12px; color: gray; background-color:#D1ECF1\">
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius:7px 7px 0 0\">Median attendance</td>
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius:7px 7px 0 0\">Median grade</td>
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius:7px 7px 0 0\">Median ontime</td>
            </tr>
            <tr style=\"font-size: 25px; color: gray; background-color:#D1ECF1\">
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius: 0 0 7px 7px; width:34%\">{:.2f}%</td>
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius: 0 0 7px 7px; width:32%\">{:.2f}</td>
                <td style=\"border:2px solid white; border-top-style:hidden; border-radius: 0 0 7px 7px; width:34%\">{:.2f}%</td>
            </tr>
        </table>
        '''.format(median_atten, median_grade, median_ontime),
        unsafe_allow_html=True)


import altair as alt
# from vega_datasets import data


def make_hist(series, bins):
    count, edges = np.histogram(series, bins=bins)

    hist_df = pd.DataFrame({'count': count, 'left': edges[:-1], 'right': edges[1:]})
    hist_df['interval'] = hist_df.apply(lambda row: '{:.0f} to {:.0f}'.format(row.left, row.right), axis = 1)
    hist_df.drop(columns=['left', 'right'], inplace=True)

    return hist_df


def graph_multi_hist(df, x_field, color_field, x_title='Value', y_title='Number of students', width=600, height=400, max_nbins=10):
    selection = alt.selection_multi(fields=[color_field])
    color = alt.condition(selection, alt.Color('%s:N'%color_field, legend=None), alt.value('lightgray'))
    opacity = alt.condition(selection, alt.value(0.8), alt.value(0.2))

    hist = alt.Chart(df).mark_area(interpolate='step').encode(
                x=alt.X('%s:Q'%x_field, bin=alt.Bin(maxbins=max_nbins), title=x_title),
                y=alt.Y('count()', stack=None, title=y_title),
                color=color,
                opacity=opacity,
                ).properties(width=width, height=height)

    legend = alt.Chart(df).mark_bar().encode(
                y=alt.Y('%s:N'%color_field, title=None, axis=alt.Axis(orient='right')),
                color=color,
                opacity=opacity
                ).add_selection(selection)

    st.altair_chart((hist | legend).configure_axis(titleFontSize=15))
    

def graph_hist(df, x_field, x_title='Value', y_title='Number of students', width=635, height=400, max_nbins=10):
    hist = alt.Chart(df).mark_bar(opacity=0.8).encode(
                x=alt.X("%s:Q"%x_field, title=x_title, bin=alt.Bin(maxbins=max_nbins)), 
                y=alt.Y('count()', title=y_title),
                tooltip=['count()']
                ).configure_axis(titleFontSize=15
                ).properties(width=width, height=height).interactive()

    st.altair_chart(hist)


def graph_submission(df):
    chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('percentage', stack='normalize', axis=alt.Axis(format='%')),
                y=alt.Y('name', title=''),
                color=alt.Color('status', scale=alt.Scale(domain=['ontime', 'late', 'missed'],
                                                          range=['#54a24b', '#eeca3b', '#e45756'])),
                tooltip=['name', 'status', 'percentage']
                ).properties(width=700, height=700
                ).configure_axis(labelFontSize=15, titleFontSize=15
                ).configure_legend(labelFontSize=12)

    st.altair_chart(chart)


def graph_LMS_time(df):
    chart = alt.Chart(df).mark_bar().encode(
            x='amount:Q', 
            y=alt.Y('type:O', title='', axis=alt.Axis(labels=False)),
            color=alt.Color('type:N', title=None),
            row= alt.Row('name:N', title='', spacing=5, header=alt.Header(labelAngle=0, labelAlign='left', labelFontSize=15)),
            tooltip=['name', 'type', 'amount']
            ).properties(width=470, height=20
            ).configure_axis(titleFontSize=15
            ).configure_legend(labelFontSize=12)

    # add mean lines?

    st.altair_chart(chart)


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Label
from bokeh.models.glyphs import Text
from bokeh.palettes import Category10
from bokeh.transform import cumsum
from math import pi


def graph_donut(percentage_df, title, center_text='', value_fname='value', percentage_fname='percentage'):
    percentage_df['angle'] = percentage_df['percentage']/100 * 2 * pi

    palette = {3: ['#54a24b', '#eeca3b', '#e45756'], # for plotting submission
               4: ['#4c78a8', '#54a24b', '#eeca3b', '#e45756']} # for LMS mess distribution 
    try:
        percentage_df['color'] = palette[len(percentage_df.index)]
    except KeyError:
        percentage_df['color'] = list(Category10[len(percentage_df.index)]) # for others

    fig = figure(plot_height=400, plot_width=700, title=title, x_range=(-.5, .5))

    fig.annular_wedge(x=0, y=1, inner_radius=0.15, outer_radius=0.25, direction="anticlock",
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                    hover_fill_alpha = 1.0, hover_fill_color = 'skyblue', line_color="white", 
                    legend_field=value_fname, color='color', alpha=0.9, source=percentage_df, name='donut')

    tooltips=[(value_fname, "@%s"%value_fname), (percentage_fname, "@%s"%percentage_fname+"{0.2f}%")]
    hover = HoverTool(tooltips=tooltips, names=['donut'])
    fig.add_tools(hover)

    text_source = ColumnDataSource(dict(x=[0], y=[1], text=[center_text]))
    text_glyph = Text(x="x", y="y", text='text', text_font_size='18pt', text_font='IBM Plex Sans', 
                      text_color='gray', text_align='center', text_baseline='middle')
    fig.add_glyph(text_source, text_glyph)

    fig.axis.axis_label=None
    fig.axis.visible=False
    fig.grid.grid_line_color = None

    fig.title.align = 'center'
    fig.title.text_font_size = '12pt'

    st.bokeh_chart(fig)


#=======================================================================
# ADVANCED - ML METHODS:

from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf


def normalize(course_df, weights_dict):
    # normalize the number of assessments by taking the mean of each type:
    for ass in weights_dict.keys(): 
        course_df[ass] = course_df.filter(regex='^{}'.format(ass)).mean(axis=1)


def reweigh(course_df, weights_dict):
    # reweigh final grade:
    course_df['final_grade'] = 0
    for ass, w in weights_dict.items():
        course_df['final_grade'] += course_df[ass]*(w/100)


def get_letter_gr(course_df, D_cutoff, step):
    grade_bins = [0] + [l-1 for l in range(D_cutoff, D_cutoff+4*step, step)] + [100]
    letters = ['F', 'D', 'C', 'B', 'A']
    return pd.cut(course_df.final_grade, grade_bins, labels=letters) 


def knn(cur_course_df, pre_courses_df, selected_vars):
    st.sidebar.header('Control Center')
    st.sidebar.warning('We only consider A, B, C, D, and F.')
    D_cutoff = st.sidebar.number_input("Cutoff point of a D:", min_value=1, max_value=96, value=60)
    step = st.sidebar.number_input("Step:", min_value=1, max_value=20, value=10)
    K = st.sidebar.slider("Select K:", min_value=1, max_value=10, value=5) # need to change this: max = len//5
 
    if st.button('Run model'):
        pre_courses_df.reset_index(inplace=True)
        pre_courses_df['letter_gr'] = get_letter_gr(pre_courses_df, D_cutoff, step)

        X = pre_courses_df[selected_vars].copy()
        Y = pre_courses_df['letter_gr'].copy()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        knn = neighbors.KNeighborsClassifier(n_neighbors=K, weights='uniform') # what's the arg weights for?
        knn.fit(X_train, Y_train)

        test_prediction = knn.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, test_prediction)

        cur_X = cur_course_df[selected_vars].copy()
        cur_prediction = knn.predict(cur_X).copy()

        prediction_df = pd.DataFrame({'name': cur_course_df['f_name'] + ' ' + cur_course_df['l_name']})
        prediction_df['predicted_grade'] = cur_prediction
        prediction_df.set_index('name', inplace=True)

        summary_df = prediction_df.predicted_grade.value_counts()
        summary_df.rename('predicted_count', inplace=True)
        
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('**Accuracy:** {:.2%}'.format(accuracy))
        st.table(summary_df)
        st.table(prediction_df)
    

def ols_reg(cur_course_df, pre_courses_df, selected_vars):
    formula = 'final_grade ~ {}'.format(selected_vars[0])
    for v in selected_vars[1:]:
        if v=='race' or v=='gender':
            formula += ' + C({})'.format(v) # treat as categorical       
        else:
            formula += ' + {}'.format(v)
    
    if st.button('Run model'):
        model = smf.ols(formula, data=pre_courses_df) 
        results = model.fit()

        prediction_df = pd.DataFrame({'name': cur_course_df['f_name'] + ' ' + cur_course_df['l_name']})
        prediction_df['predicted_grade'] = results.predict(cur_course_df[selected_vars])
        prediction_df.set_index('name', inplace=True)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('$$R^2$$: {:.2f}'.format(results.rsquared))
        st.markdown('$$p$$-value: {:.2f}'.format(results.f_pvalue))
        st.table(prediction_df)


def multino_reg(cur_course_df, pre_courses_df, selected_vars):
    st.sidebar.header('Control Center')
    st.sidebar.warning('We only consider A, B, C, D, and F.')
    D_cutoff = st.sidebar.number_input("Cutoff point of a D:", min_value=1, max_value=96, value=60)
    step = st.sidebar.number_input("Step:", min_value=1, max_value=20, value=10)
 
    if st.button('Run model'):
        pre_courses_df['letter_gr'] = get_letter_gr(pre_courses_df, D_cutoff, step)
        # encode letter_gr:
        le = LabelEncoder()
        pre_courses_df['letter_gr'] = le.fit_transform(pre_courses_df['letter_gr'])
        consider_data = pre_courses_df[selected_vars+['letter_gr']] 

        formula = 'letter_gr ~ {}'.format(selected_vars[0])
        for v in selected_vars[1:]:
            if v=='race' or v=='gender':
                formula += ' + C({})'.format(v) # treat as categorical       
            else:
                formula += ' + {}'.format(v)

        try:
            model = smf.mnlogit(formula, data=consider_data)
            results = model.fit_regularized()
            
            prediction_df = pd.DataFrame({'name': cur_course_df['f_name'] + ' ' + cur_course_df['l_name']})
            prediction = results.predict(cur_course_df[selected_vars])
            prediction_df = prediction_df.join(prediction)
            prediction_df.set_index('name', inplace=True)
            
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('Pseudo $$R^2$$: {:.2f}'.format(results.prsquared))
            # decode letter_gr:
            prediction_df.columns = [le.inverse_transform(prediction.columns)]
            # switch to percentage format and display:
            st.table(prediction_df.applymap(lambda x: '{:.2%}'.format(x)))
        
        except np.linalg.LinAlgError:
            non_unique_vars = consider_data.columns[consider_data.nunique() <= 1]
            non_unique_vars = " ".join(tuple(non_unique_vars))
            st.error(
            '''
            The following variable(s) are non-unique: {}.\n
            They have a singular value for all students. Please deselect these variables.
            '''.format(non_unique_vars))


if __name__ == '__main__':
    s = pd.Series([1, 2, 1, 0, 8, 5, 5])
    pdf = estimate_pdf(s)