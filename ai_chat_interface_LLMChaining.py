import os
from dotenv import dotenv_values

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ai_chat_interface_utils import extracting_results
from ai_chat_interface_DataManipulation import PandasDataFrameDataManipulation
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain



class LLMChaining:

    def __init__(self, data_manipulation : PandasDataFrameDataManipulation):
        
        # Intialized with PandasDataFrameDataManipulation object
        self.data_manipulation = data_manipulation
        self.column_names = list(data_manipulation.dataFrame.columns)

        # Setting the environment variable, the Hugging Face Token
        # It is stored in .env file and should be hidden
        os.environ["HF_TOKEN"] = dotenv_values('.env')["HF_TOKEN"]

        # The model name, downloaded from Hugging Face
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"

        # Creating the model
        offload_folder = "./offload_folder"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", offload_folder=offload_folder)

        # Create a Hugging Face pipeline
        self.pipe = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            max_new_tokens=80,
            max_length=500, 
            temperature=0.0005, 
            truncation=True)
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # self.functions and self.functions_dict contain the explanations of what PandasDataFrameDataManipulation
        # do and what there input should be, respectively. These are necessary in order for LLM to understand there use
        self.functions = self.data_manipulation.functions
        self.functions_dict = self.data_manipulation.functions_dict




        # PromptTemplate for choosing a function - chained
        template = f"""
        We have the columns named:
        {self.column_names}
        Based on the text, choose the most suitable function out of the following ones:
        {self.functions}"""+"""
        Text: {input_text}
        Write only the name of the function chosen as the most suitable with empty () as the result.
        """
        self.prompt_functionChoosingChained = PromptTemplate(
            input_variables=["input_text"],
            template=template,
        )

        # PromptTemplate for choosing a function - chained - extraction
        template = """
        From the input text extract what the final result was. The final result should have the explanation. If there are multiple different answers take the first one for which the choice has an explanation.
        Text: {input_text}
        Write only the name of the function chosen as the most suitable with empty () as the result inbetween these strings %start and %end.
        """
        self.prompt_functionChoosingChainedExtraction = PromptTemplate(
            input_variables=["input_text"],
            template=template,
        )

        # chaining the Prompts for function choosing
        self.chain_functionChoosingChained = LLMChain(
            llm=self.llm, 
            prompt=self.prompt_functionChoosingChained)
        self.chain_functionChoosingChainedExtraction = LLMChain(
            llm=self.llm, 
            prompt=self.prompt_functionChoosingChainedExtraction)
        self.chain_full_functionChoosing = SimpleSequentialChain(
            chains=[self.chain_functionChoosingChained, self.chain_functionChoosingChainedExtraction], 
            verbose=True)
        
        # PromptTemplate for generating a command - chained
        template = f"""
        We have the columns named:
        {self.column_names}
        Based on the text, and the chosen function, generate the input to match the function:
        """+"""
        Text: {input_text}
        Write the desired result as a call to the chosen function.
        """
        self.prompt_commandGenerationChained = PromptTemplate(
            input_variables=["input_text"],
            template=template,
        )

        # PromptTemplate for generating a command - chained - extraction
        template = """
        From the input text extract what the final result was. The final result should have the explanation. If there are multiple different answers take the first one for which the choice has an explanation.
        Text: {input_text}
        Write the desired result as a call to the chosen function inbetween these strings %start and %end.
        """
        self.prompt_commandGenerationChainedExtraction = PromptTemplate(
            input_variables=["input_text"],
            template=template,
        )

        # chaining the Prompts for command generating
        self.chain_commandGenerationChained = LLMChain(
            llm=self.llm, 
            prompt=self.prompt_commandGenerationChained)
        self.chain_commandGenerationChainedExtraction = LLMChain(
            llm=self.llm, 
            prompt=self.prompt_commandGenerationChainedExtraction)
        self.chain_full_commandGeneration = SimpleSequentialChain(
            chains=[self.chain_commandGenerationChained, self.chain_commandGenerationChainedExtraction], 
            verbose=True)
        
    # Main function for running a user input
    def run(self, query : str):
        input_text = query
        result1 = self.chain_full_functionChoosing.run(input_text)
        input1 = self.prompt_functionChoosingChainedExtraction.format(input_text=result1)
        chosen_function = extracting_results(input1, result1)
        result2 = self.chain_full_commandGeneration.run("Chosen function: "+self.functions_dict[chosen_function]+" Text: "+input_text)
        input2 = self.prompt_commandGenerationChainedExtraction.format(input_text=result2)
        final_result = extracting_results(input2, result2)
        print(exec("self.data_manipulation." + final_result))