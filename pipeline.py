
import pandas as pd

def load_data(loadpath):
    #reads a csv and return the dataframe
    df = pd.read_csv(loadpath)
    return df

    

def remove_missing(df):
    #drops all rows with missing values
    #null = df.isnull().sum()
    #dropna() will automatically delete the nul values
    return df.dropna()

    

def show_summary(df):
    #prints shape and the first five rows
    print("Titanic Shape:(columns ,rows)" ,df.shape)
    print(df.head(5))


l = load_data("train.csv")
c = remove_missing(l)
show_summary(c)

    
'''  
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_missing(df):
    return df.dropna()

def filter_adults(df, age_column, min_age=18):
    return df[df[age_column] > min_age]

def clean_income(df, income_column):
    df[income_column] = df[income_column].str.replace(",", "").astype(float)
    return df

def run_pipeline(filepath):
    df = load_data(filepath)
    df = remove_missing(df)
    df = filter_adults(df, "age")
    df = clean_income(df, "income")
    print(f"Pipeline complete. Shape: {df.shape}")
    return df

'''