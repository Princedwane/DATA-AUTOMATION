
import pandas as pd

def load_data(loadpath):
    """
    The function reads a CSV file and returns the Dataframe

    parameters:
        loadpath(str): the function excpects a string which is the actusl path or name of the file

    Returns:
        a data frame is returned  

    """
    try:
        df = pd.read_csv(loadpath)
        print(f"Date is loaded sucessfully.Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{loadpath} not found.")
        return None


    

def remove_missing(df):
    
    """
    The function: removes the missing values found in the file.

    Parameters: data frame recived and using the dropna() to automatically remove all empty values

    Returns:
        1-total number of missing values
        2- a clean data with the number of complete columns and rows

    """
    total_nulls= df.isnull().sum().sum()

    print(f"Total Nulls: {total_nulls}")

    if total_nulls >0:
        print("Nulls deleted:", total_nulls)
        return df.dropna()
    else:
        print("No missing values were actually found!")
        return df


    

def show_summary(df):
    """
    The function: Prints the summary of the data

    Parameters:
        Receives the clean df 
        Head(5) to print the first 5  elements(rows)
        shape to print the overall clean shape aftrer removing the nulls

    """
    #prints shape and the first five rows
    print("Titanic Shape:(columns ,rows)" ,df.shape)
    print(df.head(5))

if __name__ == "__main__":
    l = load_data("train.csv")
    if l is not None:
        c = remove_missing(l)
        show_summary(c)

#Day 5, REFACTORING THE CODE #
#ADDING DOCSTRINGS -- descriptioon of what the method does
#ERROR HANDLING : like what happen if file dont exist
#MAIN BLOCK: proffesional way of running the code

