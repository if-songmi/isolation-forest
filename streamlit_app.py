import streamlit as st
from streamlit_javascript import st_javascript
import time

st.subheader("Isolation Forest", divider='gray')
while True:
    PassingData = st_javascript('parent.window.PassingData')
    if PassingData:
        print(PassingData);
        break
    time.sleep(1)

