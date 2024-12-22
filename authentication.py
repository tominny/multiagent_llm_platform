"""
authentication.py - Contains functions for user login, logout, and session management.
"""

import streamlit as st
from passlib.hash import bcrypt
from db import get_user

def login_user(username: str, password: str) -> bool:
    """
    Validates the username/password against the database.
    If valid, sets session state accordingly. Returns True if login is successful.
    """
    user_data = get_user(username)
    if user_data is None:
        return False

    user_id, db_username, db_password_hash = user_data

    if bcrypt.verify(password, db_password_hash):
        st.session_state["logged_in"] = True
        st.session_state["username"] = db_username
        st.session_state["user_id"] = user_id
        return True
    return False

def logout_user():
    """
    Logs out the current user by clearing session state.
    """
    if "logged_in" in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["user_id"] = None

def is_user_logged_in() -> bool:
    """
    Returns True if the user is currently logged in, otherwise False.
    """
    return st.session_state.get("logged_in", False)

def get_current_user():
    """
    Returns (user_id, username) from session state.
    """
    return (
        st.session_state.get("user_id"),
        st.session_state.get("username")
    )
