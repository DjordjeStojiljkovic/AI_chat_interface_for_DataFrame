# AI Chat Interface for Data Analysis using `pandas.DataFrame`

This small project is used to integrate LLM model in order to extract information from `pandas.DataFrame` by simple user input to the LLM model.
This project is constructed as an example and is intended as an entry test for Internship at JetBrains.

## Structure

Structure of the project consists of 2 main layers:
1) Set of predefined commands for manipulating `pandas.DataFrame` data
2) AI chat that is used to generate sequences of such commands

Layer 1. is implemented in `ai_chat_interface_DataManipulation.py` as `PandasDataFrameDataManipulation` class. It currently has these functions:
- `column_select()` - used for selecting a specific column of the `pandas.DataFrame`
- `predicate_search()` - used for searching the data that satisfies the given condition
The project is constructed so it can be extended with more functions.

Layer 2. is implemented in `ai_chat_interface_LLMChaining.py` through `LLMChaining` class using `langchain` library. It has 2 levels:
- First layer is used for determining which data manipulation function should be called and is a chain of 2 sequential `langchain.chains.LLMChain`
- Second layer is used for establishing the required input to the chosen command and also consists of 2 chains.

## Model

Code is written to support the model https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct. Downloading this model requires the permission from Meta, and one would need the Hugging Face profile and  Hugging Face API token. This can be acquired easily. Many other Hugging Face models can be used as well, for example OpenAI modelsm through `langchain`. In that case apropriate permissions and API keys if necesseary should be acquired.

## Use

Various dependecies are listed in `requirements.txt` file.
In the main file `ai_chat_interface_for_DataFrame.py` is a main function in which initialization is demonstrated. A `pandas.DataFrame` is initiated from a `.csv` file. Example file `deniro.csv` was downloaded from https://people.sc.fsu.edu/~jburkardt/data/csv/csv.html . User can enter simple inputs from the command line like 'select all movies with score>50' and program will return the requested data.