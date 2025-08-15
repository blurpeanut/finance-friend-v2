import pandas as pd

# Read the department data from Excel
df1 = pd.read_excel('data/department data.xlsx')

# Function to calculate allocated budget
def calculate_allocated_budget(df1, budget_caps):
    for budget_type, cap in budget_caps.items():
        df1[budget_type] = df1["Headcount"] * cap
    return df1[["Departments", "Headcount"] + list(budget_caps.keys())]

# Function to calculate vote utilisation
def calculate_vote_utilisation():
    df = pd.read_excel("data/sample data.xlsx")
    
    # Strip column names and store a copy of original column list
    df.columns = df.columns.str.strip()
    original_columns = df.columns.tolist()

    # Convert amount to float (remove $ and commas)
    df["Amount Used"] = df["Amount Used"].replace('[\$,]', '', regex=True).astype(float)

    # Group and reshape
    grouped = df.groupby(["Directorate", "Vote Type"])["Amount Used"].sum().reset_index()
    pivot = grouped.pivot(index="Directorate", columns="Vote Type", values="Amount Used").fillna(0).reset_index()

    return pivot




