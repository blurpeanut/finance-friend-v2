import os, hmac, streamlit as st

def check_password() -> bool:
    """Return True if user entered the correct password."""
    # 1) Read configured password safely (from secrets or env)
    configured_pwd = st.secrets.get("password") or os.getenv("ADMIN_PASSWORD", "")
    if not configured_pwd:
        st.warning("No admin password is configured. Set st.secrets['password'] or ADMIN_PASSWORD.")
        # Decide your policy: allow access for now or block.
        # return True  # <- Uncomment to allow access when not configured
        return False   # <- Safer default: block access when not configured

    # 2) Initialize session keys
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if "password" not in st.session_state:
        st.session_state.password = ""

    # 3) If already authenticated, allow through
    if st.session_state.password_correct:
        return True

    # 4) Check handler (use .get to avoid KeyError)
    def _password_entered():
        user_pwd = st.session_state.get("password", "")
        st.session_state.password_correct = hmac.compare_digest(user_pwd, configured_pwd)
        # Don’t keep the raw password in memory
        st.session_state.pop("password", None)

    # 5) UI
    st.text_input("Password", type="password", key="password")
    st.button("Log in", on_click=_password_entered)

    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("😕 Password incorrect")

    return False