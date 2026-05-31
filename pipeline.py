
import pandas as pd
import mlflow

mlflow.set_experiment("Titanic_Pipeline")

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
        print(f"Data is loaded sucessfully.Shape: {df.shape}")
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

def train_model(df):
    """
    Trains a Logistic Regression model on the Titanic dataset.
    Parameters:
        df: the clean dataframe received from remove_missing
    Returns:
        the trained model and its accuracy score
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = df[["Pclass", "Age", "Fare"]]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #train test splits data into 2 parts, 80%for training model
    #20% for testing how well it learned
    #test_size =0.2 mean 20% foes to testing 
     
    model = LogisticRegression()
    model.fit(X_train, y_train) #where the actual learning happen

    predictions = model.predict(X_test) #model predicts on unseen data
    accuracy = accuracy_score(y_test, predictions) #comparres rpredictions against real answers and provide percentage of accuracy

    return model, accuracy # returns the trained model and its accuracy scoreh

if __name__ == "__main__":
    l = load_data("train.csv")
    if l is not None:
        with mlflow.start_run():
            c = remove_missing(l)

            mlflow.log_metric("rows_before", len(l))
            mlflow.log_metric("rows_after", len(c))
            mlflow.log_metric("rows_removed", len(l) - len(c))
            
            model , accuracy = train_model(c)

            mlflow.log_param("model_type", "LogisticRegression") #logs the setting 
            mlflow.log_param("test_size", 0.2)
            mlflow.log_metric("accuracy", accuracy)#logs the accuracy number

            mlflow.sklearn.log_model(model , "model") #saving the actual trained modelinside mlflow

            show_summary(c)
            print(f"Model Accuracy: {accuracy}")

            
          
#Day 5, REFACTORING THE CODE #
#ADDING DOCSTRINGS -- descriptioon of what the method does
#ERROR HANDLING : like what happen if file dont exist
#MAIN BLOCK: proffesional way of running the code


