import streamlit as st

st.markdown("## About Us - Finance Friend")

st.markdown("""
#### Making Essential Finance Data Accessible — Without the Back-and-Forth

In many organisations, key finance data like budget utilisation or policy clarifications often live in scattered emails, buried PDFs, or slow-to-load SharePoint files. Officers looking to manage their budget or clarify a vote policy may have to wait on Finance teams — even for simple questions.

**Finance Friend** was built to change that.

This self-service tool provides officers with a faster, clearer way to understand decentralised vote usage and get answers to finance queries — all in one place.
""")

st.markdown("#### 🎯 What This Project Aims to Solve")
st.markdown("""
This project was developed under the AI Champions Bootcamp to explore how retrieval-augmented generation (RAG) and other AI techniques can streamline internal finance processes. This project is a proof-of-concept to show how AI can bridge the gap between central Finance data and end-user needs.
""")

st.markdown("#### 🧩 Core Objectives and Benefits")
st.markdown("""
**1. Seamless Data Pipeline**  
- Automated ingestion of GL files or Excel sheets  
- Built-in data cleaning and standardisation (amounts, vote types, dates, categories)  
- Accessible via a single link — no account setup needed  
💡 *Result: Clean, structured data instantly accessible across directorates.*

**2. Instant Clarity Through Visualisation**  
- Interactive 100% stacked bar charts to show utilisation vs budget caps  
- Visual alerts for overspending (e.g. red bars for exceeded budgets)  
- Filters by vote type and directorate for targeted views  
💡 *Result: Officers can immediately see where they stand, with zero guesswork.*

**3. AI-Powered Support**  
- Built-in chatbot trained on vote-related documents and FAQs  
- Natural language Q&A like “Can I claim this under Staff Welfare?”  
💡 *Result: Instant answers, fewer emails to Finance.*
""")

st.markdown("#### ⚙️ Key Features")
st.markdown("""
- **Vote Utilisation Dashboard**  
  An interactive bar chart that shows how much of each vote category has been used — filterable by directorate for targeted insights.
            
- **Downloadable Transaction Table**    
  A CSV file with all transactions, including remarks for easy offline edits.

- **AI-Powered Chat Assistant ("VoteBot")**  
  Users can ask natural-language questions about finance policies (e.g. “Can I claim this under staff welfare?”) and get instant answers sourced from uploaded policy documents.
""")

st.markdown("#### 🗂️ Data Sources")
st.markdown("""
- **Finance GL Drill**: Transactions from the NFS system  
- **Reference Documents**: PDFs on decentralised votes and policy circulars   
""")

