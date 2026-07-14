import streamlit as st

st.set_page_config(
    page_title="AI Task Management System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Task Management System")

st.markdown("---")

st.header("Welcome")

st.write(
    """
    This dashboard predicts:

    ✅ Task Category

    ✅ Task Priority

    ✅ Recommended Employee
    """
)

st.success("Week 4 Dashboard Started Successfully!")