import time

import numpy as np
import streamlit as st

from globals import *

"""
# Lifelong MAPF
"""
with st.form("my_form"):
    iterations = st.slider('How many iters?', 0, 130, 3)

    submitted = st.form_submit_button("Submit")

if submitted:
    fig, ax = plt.subplots()
    st_fig = st.pyplot(fig)
    my_bar = st.progress(0, text='progress...')

    x_list, y_list = [], []
    for i in range(iterations + 1):
        x_list.append(i)
        y_list.append(np.sin(i))

        # plot
        ax.cla()
        ax.plot(x_list, y_list)
        ax.set_xlim(0, iterations)
        st_fig.pyplot(fig)
        my_bar.progress(i / iterations, text=f'{int(i / iterations * 100)}%')
        time.sleep(1)

    '''
    ## after run
    '''

'''
## after
'''
