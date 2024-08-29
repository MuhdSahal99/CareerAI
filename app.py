import streamlit as st
from PIL import Image
import os
from pages import candidate_page, employer_page
import nltk

import nltk
nltk.download('punkt_tab')

def main():

    # Set page configuration
    st.set_page_config(page_title="TalentAlign", layout="wide")

    # Create two columns
    col1, col2 = st.columns([1, 3])

    with col1:
        # Load and display the logo in the sidebar
        logo_path = os.path.join("assets", "logo.png")
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.sidebar.image(logo, width=250)  # Adjust width as needed
        else:
            st.sidebar.warning("Logo file not found. Please add logo.png to the assets folder.")

        # Add sidebar content
        st.sidebar.title("TalentAlign")
        page = st.sidebar.radio("Select Page", ["Candidate", "Employer"])

    with col2:
        # Main content area
        if page == "Candidate":
            candidate_page.show()
        elif page == "Employer":
            employer_page.show()

if __name__ == "__main__":
    main()
