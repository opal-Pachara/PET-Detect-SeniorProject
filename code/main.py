import streamlit as st
from code import data_model  # แก้การ import ให้ถูกต้อง

# ------------------------------
# Sidebar menu
# ------------------------------
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Home", "Data"])

# ------------------------------
# Page rendering
# ------------------------------
def main():
    if page == "Home":
        st.title("Hello world!")
        st.markdown("Dashboard Source Code !!!")

    elif page == "Data":
        data_model()

# ------------------------------
# Run the app
# ------------------------------
if __name__ == "__main__":
    main()
