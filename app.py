"""
app.py - Main Streamlit application:
  - User login/logout/signup
  - Multi-agent USMLE vignette generation
  - Database persistence and retrieval
"""

import streamlit as st
from db import init_db, create_user, save_vignette, get_user_vignettes
from authentication import login_user, logout_user, is_user_logged_in, get_current_user
from openai_utils import generate_usmle_vignette

def main():
    # Initialize the DB (create tables if needed)
    init_db()

    # Ensure session keys
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["user_id"] = None

    # Navigation Menu
    menu = ["Login", "Signup", "Generate Vignette", "My Vignettes", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        show_login_page()
    elif choice == "Signup":
        show_signup_page()
    elif choice == "Generate Vignette":
        if is_user_logged_in():
            show_generate_vignette_page()
        else:
            st.warning("Please log in to generate vignettes.")
    elif choice == "My Vignettes":
        if is_user_logged_in():
            show_user_vignettes_page()
        else:
            st.warning("Please log in to view your vignettes.")
    elif choice == "Logout":
        if is_user_logged_in():
            logout_user()
            st.success("You have been logged out.")
        else:
            st.warning("You are not logged in.")


def show_login_page():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")


def show_signup_page():
    st.header("Signup")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")

    if st.button("Signup"):
        if not new_username or not new_password:
            st.error("Please enter both username and password.")
        else:
            if create_user(new_username, new_password):
                st.success("User created successfully! You can now log in.")
            else:
                st.error("Username already exists. Please choose a different one.")


def show_generate_vignette_page():
    st.header("Generate a USMLE-Style Clinical Vignette (Multi-Agent)")
    topic = st.text_input("Enter a topic (e.g., Multiple Sclerosis, Parkinson's Disease)")

    if st.button("Generate Vignette"):
        if not topic:
            st.warning("Please enter a topic before generating.")
            return

        st.info("Generating a multi-agent vignette. Please wait...")

        # Generate the vignette
        init_vignette, final_vignette, conversation_json = generate_usmle_vignette(topic)

        # Display
        st.subheader("Initial Vignette (From Vignette-Maker)")
        st.text_area("Initial Vignette", init_vignette, height=200)

        st.subheader("Final Vignette (From Show-Vignette)")
        st.text_area("Final Vignette", final_vignette, height=200)

        st.subheader("Conversation JSON")
        st.text_area("Conversation", conversation_json, height=300)

        # Save to DB
        user_id, _ = get_current_user()
        save_vignette(user_id, topic, init_vignette, final_vignette, conversation_json)
        st.success("Vignette saved to your account!")


def show_user_vignettes_page():
    st.header("My Generated Vignettes")
    user_id, _ = get_current_user()
    vignettes = get_user_vignettes(user_id)

    if not vignettes:
        st.info("No vignettes found. Generate one first!")
        return

    for (vignette_id, topic, init_vig, final_vig, convo) in vignettes:
        # Simple display with no nested expanders
        st.subheader(f"Vignette #{vignette_id} - Topic: {topic}")
        st.write("**Initial Vignette:**")
        st.write(init_vig)
        st.write("**Final Vignette:**")
        st.write(final_vig)
        st.write("**Conversation JSON:**")
        st.text_area("Conversation", convo, height=200)
        st.markdown("---")  # Divider between vignettes


if __name__ == "__main__":
    st.set_page_config(page_title="USMLE Vignette Generator (Multi-Agent)", layout="wide")
    main()
