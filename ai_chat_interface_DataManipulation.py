import pandas as pd

class PandasDataFrameDataManipulation:
    
    def __init__(self, file_path : str):
        self.dataFrame = pd.read_csv(file_path, on_bad_lines='skip')

        # self.functions and self.functions_dict contain the explanations of what PandasDataFrameDataManipulation
        # do and what there input should be, respectively. These are necessary in order for LLM to understand there use
        # In case of extending the project with more functions, these explanations should be added to these variables
        self.functions = """
        column_select() : (selecting all rows from the named column, does not need conditions)
        predicate_search() : (search the data based on a condition, takes only rows that satisfy the condition)
        """
        self.functions_dict = {
            "column_select()" : "column_select(column_to_select : str) : (input should be just the name of the column)",
            "predicate_search()" : "predicate_search(condition : str) : (input should be conditions in mathematical notation that are joined by the keyword and if there are multiple conditions)"
        }
    
    # setting the inner DataFrame
    def set_dataFrame(self, dataFrame : pd.DataFrame) -> None:
        self.dataFrame = dataFrame
    
    # function for searching the DataFrame based on a condition
    def predicate_search(self, predicate_query : str) -> None:
        print(self.dataFrame.query(predicate_query))
    
    # function for searching a specific column
    def column_select(self, column_to_search : str) -> None:
        print(self.dataFrame[column_to_search])