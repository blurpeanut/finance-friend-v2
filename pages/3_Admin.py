import os
from dotenv import load_dotenv
import streamlit as st

# Load .env file
load_dotenv()

# Get admin password from environment
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Set up folders to store uploaded files
os.makedirs("uploaded_gl_files", exist_ok=True)
os.makedirs("uploaded_policy_docs", exist_ok=True)

st.title("🛠️ Admin Page")

# Ask for password
password = st.text_input("Enter admin password:", type="password")
if password != ADMIN_PASSWORD:
    st.warning("Access denied. Please enter the correct password.")
    st.stop()

# Upload GL file
st.subheader("📥 Upload GL Data File")
gl_file = st.file_uploader("Upload **Latest** GL Excel or CSV file", type=["xlsx", "csv"])
if gl_file:
    # Overwrite the file used in your app
    target_path = os.path.join("data", "sample data.xlsx")  # Keeping the name fixed
    with open(target_path, "wb") as f:
        f.write(gl_file.read())
    st.success("✅ GL data file updated. App will now use the latest uploaded file.")

# Upload policy document
st.subheader("📄 Upload Policy Document")
policy_file = st.file_uploader("Upload policy file (PDF or TXT)", type=["pdf", "txt"])
if policy_file:
    policy_path = os.path.join("uploaded_policy_docs", policy_file.name)
    with open(policy_path, "wb") as f:
        f.write(policy_file.read())
    st.success(f"✅ Policy document uploaded: {policy_file.name}")