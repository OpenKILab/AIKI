import streamlit as st
from streamlit import session_state as ss
if "btn_state" not in ss: ss.btn_state = False
if st.button('获取'):
  ss.btn_state = True
if ss.btn_state:
  lists = ['a', 'b', 'c', 'd', 'e']
  selected_opt = st.selectbox('列表', lists, index=None)
  if selected_opt: st.write('选择的值：' + selected_opt)
