import pandas as pd
from ai_chat_interface_DataManipulation import PandasDataFrameDataManipulation
from ai_chat_interface_LLMChaining import LLMChaining

if __name__ == "__main__":

    # File path to do data in .csv
    file_path = "deniro.csv"

    # Creating PandasDataFrameDataManipulation object for getting outputs from DataFrame
    data_manipulation = PandasDataFrameDataManipulation(file_path=file_path)

    # Creating LLMChaining object, custom class for acquiring and communicating with the LLM model
    llm_chaining = LLMChaining(data_manipulation=data_manipulation)

    # Print out the DataFrame
    print(data_manipulation.dataFrame)

    # Loop for inputs
    while True:
        input_text = input()
        if input_text == 'end':
            break
        llm_chaining.run(input_text)





