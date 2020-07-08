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
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CheckboxGroup
from bokeh.layouts import column, row, WidgetBox


'''
def graph_hist(series, title, x_label, y_label='Number', width=600, height=500):
    hi = series.max()
    lo = series.min()
    hist, edges = np.histogram(series, bins=(hi-lo)//2) # group by every 2 point

    # Put the information in a dataframe
    hist_df = pd.DataFrame({'hist': hist, 'left': edges[:-1], 'right': edges[1:]})    
    # Add a column that specifies the interval
    hist_df['interval'] = ['{:.0f} to {:.0f}'.format(left, right) for left, right in zip(hist_df['left'], hist_df['right'])]
    
    # Convert dataframe to column data source
    hist_cd_src = ColumnDataSource(hist_df)

    hover = [('Range', '@interval'), ('Number', '@hist')]

    checkbox_group = CheckboxGroup(labels=["Option 1", "Option 2", "Option 3"], active=[0, 1])

    # Create a blank figure with labels:
    hist_fig = figure(plot_width=width, plot_height=height, title=title, 
                      x_axis_label=x_label, y_axis_label=y_label, tooltips=hover)

    # Add a quad glyph:
    hist_fig.quad(source=hist_cd_src, bottom=0, top='hist', left='left', right='right', 
                  fill_color='navy', line_color='white', alpha=0.5, 
                  hover_fill_alpha = 1.0, hover_fill_color = 'skyblue')

    layout = row(hist_fig, checkbox_group)
    
    return layout
'''

def style(fig):
    # Title 
    fig.title.align = 'center'
    fig.title.text_font_size = '12pt'

    # Axis titles
    #fig.xaxis.axis_label_text_font_size = '10pt'
    fig.xaxis.axis_label_text_font_style = 'bold'
    #fig.yaxis.axis_label_text_font_size = '10pt'
    fig.yaxis.axis_label_text_font_style = 'bold'

    # Tick labels
    #fig.xaxis.major_label_text_font_size = '8pt'
    #fig.yaxis.major_label_text_font_size = '8pt'

def make_hist_cdsrc(series):
    hi = series.max()
    lo = series.min()
    hist, edges = np.histogram(series, bins=(hi-lo)//2) # group by every 2 point

    # Put the information in a dataframe
    hist_df = pd.DataFrame({'hist': hist, 'left': edges[:-1], 'right': edges[1:]})    
    # Add a column that specifies the interval
    hist_df['interval'] = ['{:.0f} to {:.0f}'.format(left, right) for left, right in zip(hist_df['left'], hist_df['right'])]
    
    # Convert dataframe to column data source
    hist_cdsrc = ColumnDataSource(hist_df)

    return hist_cdsrc

def graph_hist(df, title, x_label, y_label='Number', width=600, height=500):
    # Create a blank figure with labels:
    hist_fig = figure(plot_width=width, plot_height=height, title=title, 
                      x_axis_label=x_label, y_axis_label=y_label)

    test_selection = CheckboxGroup(labels=list(df.columns), active=[0], 
                                   margin=(30, 0, 20, 30)) # margin=(top, right, bottom, left)

    selected_tests = [test_selection.labels[i] for i in test_selection.active]
    hist_cdsrc = make_hist_cdsrc(df[selected_tests[0]])

    hist_fig.quad(source=hist_cdsrc, bottom=0, top='hist', left='left', right='right', 
                  fill_color='navy', line_color='white', alpha=0.5, 
                  hover_fill_alpha = 1.0, hover_fill_color = 'skyblue')

    hover = HoverTool(tooltips=[('Range', '@interval'), ('Number', '@hist')], mode='vline')
    hist_fig.add_tools(hover)

    style(hist_fig)

    layout = row(hist_fig, test_selection)
    
    return layout


if __name__ == '__main__':
    s = pd.Series([1, 2, 1, 0, 8, 5, 5])
    pdf = estimate_pdf(s)