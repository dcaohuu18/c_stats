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


#####
# GRAPHS:

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
from vega_datasets import data


def make_hist(series, bins):
    count, edges = np.histogram(series, bins=bins)

    hist_df = pd.DataFrame({'count': count, 'left': edges[:-1], 'right': edges[1:]})
    hist_df['interval'] = hist_df.apply(lambda row: '{:.0f} to {:.0f}'.format(row.left, row.right), axis = 1)
    hist_df.drop(columns=['left', 'right'], inplace=True)

    return hist_df


def graph_hist(df, title, x_label, y_label='Number', width=600, height=500):
    # Create a blank figure with labels:
    return


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

    st.altair_chart(chart)


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Label
from bokeh.models.glyphs import Text
from bokeh.palettes import Category10
from bokeh.transform import cumsum
from math import pi


def graph_donut(percentage_df, title, center_text='', value_fname='value', percentage_fname='percentage'):
    percentage_df['angle'] = percentage_df['percentage']/100 * 2 * pi

    try:
        percentage_df['color'] = ['#54a24b', '#eeca3b', '#e45756'] # for plotting submission
    except ValueError:
        palette = list(Category10[len(percentage_df.index)])
        palette[1], palette[2] = palette[2], palette[1] # swap orange and green
        percentage_df['color'] = palette

    fig = figure(plot_height=400, plot_width=700, title=title, x_range=(-.5, .5))

    fig.annular_wedge(x=0, y=1, inner_radius=0.15, outer_radius=0.25, direction="anticlock",
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                    hover_fill_alpha = 1.0, hover_fill_color = 'skyblue', line_color="white", 
                    legend_field=value_fname, color='color', source=percentage_df, name='donut')

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


if __name__ == '__main__':
    s = pd.Series([1, 2, 1, 0, 8, 5, 5])
    pdf = estimate_pdf(s)