import streamlit as st
import pandas as pd
from streamlit_javascript import st_javascript

st.subheader("Isolation Forest", divider='gray')
df = pd.read_json("data.json")
st.dataframe(df, hide_index = True)