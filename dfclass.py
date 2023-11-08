import sys
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
import os
import openai
import pandas as pd
import numpy as np
import ast
import json
import csv

import regex as re

encoding4 = tiktoken.encoding_for_model("gpt-4")
encoding35 = tiktoken.encoding_for_model("gpt-3.5-turbo")
embedding_encoding = tiktoken.encoding_for_model("text-embedding-ada-002")


# LLM models
gpt4 = "gpt-4"
gpt46 = "gpt-4-0613"


gpt35616k = "gpt-3.5-turbo-16k"


gpt411 = "gpt-4-1106-preview"
gpt3511 = "gpt-3.5-turbo-1106"

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
tokenizer = tiktoken.get_encoding(embedding_encoding)
import math

inf = math.inf



max_tokens = 8000

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logDirectory = '/home/ec2-user/chat/logs'
client_admin_dir = '/home/ec2-user/chat/client_admin_directory'
current_directory = '/home/ec2-user/chat'
python_scirpt_directory = '/home/ec2-user/chat/python_files'


def process_string(input_string):
    number_character_in_string = len(input_string)
    return number_character_in_string


def embedding_agent(input):
    response = openai.Embedding.create(input=input, model=embedding_model)
    return response["data"][0]["embedding"]



def llmagent(messages, function_specifier="auto", functions=None, model=gpt3511, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, max_tokens=0):
    
    try:
        params = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        if max_tokens != 0:
            params["max_tokens"] = max_tokens
        
        if functions is not None:
            params["tools"] = functions
            params["tool_choice"] = {"name": function_specifier}
        
        response = openai.ChatCompletion.create(**params)

        response_message = response["choices"][0]["message"]
        usage = response["usage"]

        if response_message.get("tool_calls"):
            print("\n\nfunction call detected.\n\n")
            print(response_message)
            print()
            return (response_message["tool_calls"], usage)
        
        else:
            response_text = response["choices"][0]["message"]["content"]
            print("\n\nGenerator call detected.\n\n")
            print("\n\n", response_text, "\n\n")
            return (response_text, usage)
        
    except Exception as e:
        print(e)


def function_response(response, available_functions):
    """
    tools or functions.
    model must be gpt411

    """
    return_values = {}
    try:
        for tool_call in response:
            
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            return_values[function_name]
            print(str(function_args))

            if function_args == {}:
                return_value = function_to_call()
                return_value = str(return_value)
                print(return_value)
            else:
                return_value = function_to_call(**function_args)
                return_value = str(return_value)
                print(return_value)

            return_values[function_name] = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": return_value,
            }

        return return_values
            
    except Exception as e:
        print(f"\nException: {e}\n")


def embedding_agent(input):
    response = openai.Embedding.create(
        input=input, 
        model=embedding_model
    )
    return response["data"][0]["embedding"]




class Generator:
    def __init__(self, name):
        self.name = name
        self.messages = []
        self.generated_text = ""
        self.content = []
        self.functions = []
        self.available_functions = {}
        self.token_count = 0
        self.model = gpt411

    def token_counter(self, functions=None):
        self.token_count = 0
        token_izer = tiktoken.get_encoding(self.model)
        for msg in self.messages:
            tokens = token_izer.encode(str(msg))
            self.token_count += len(tokens)

        if functions is not None:
            for func in functions:
                tokens = token_izer.encode(str(func))
                self.token_count += len(tokens)

        print(f"\n\nToken count: {self.token_count}\n\n")
    

    def dynamic_model_change(self):
        if self.model == gpt3511 and self.token_count > 4000:
            print("\n\nToken count exceeded for gpt-3.5-turbo. Switching to gpt-3.5-turbo-16k\n\n")
            self.model = gpt35616k

    def system(self, msg):
        self.messages[0] = {"role": "system", "content": msg}


    def run(self, msg="", function_specifier="auto", model=gpt3511, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, max_tokens=None, functions=None, extended_response=0):
        """
        The extended response needs to be manually calculated.
        This is determined by the number of chained function calls you expect the llm to make.
        (not an exact science, but an expedient solution)
        
        """
        self.generated_text = ""
        self.model = model

        if msg != "":
            self.messages.append({"role": "user", "content": msg})
            self.token_counter(functions=functions)
            self.dynamic_model_change()


        _response = llmagent(self.messages, function_specifier=function_specifier, model=self.model, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, max_tokens=max_tokens, functions=functions)

        response = _response[0]
        usage = _response[1]
        self.token_count = usage.total_tokens


        if functions is not None:

            if extended_response == 0:
                print("\nextended response not detected\n")
                returned_values = function_response(response, self.available_functions)

                for tool_call in response:
                    function_name = tool_call.function.name
                    print(function_name + " detected\n")
                    rtrnstr = "Function call: " + function_name + " = " + returned_values[function_name]["content"]
                    self.content.append(rtrnstr)
                    self.generated_text += rtrnstr + "\n"


            if extended_response == 1:
                print("\n1 extended response detected\n")
                returned_values = function_response(response, self.available_functions)
                for tool_call in response:
                    function_name = tool_call.function.name
                    print(function_name + " detected\n")
                    self.messages.append(returned_values[function_name])
                
                self.token_counter(functions=functions)
                _response = llmagent(self.messages, model=self.model, max_tokens=max_tokens)
                response = _response[0]
                self.generated_text = response

        else:
            self.content.append(response)
            self.generated_text = response


    def add_available_function(self, name, function=None):
        self.available_functions[name] = function
        print(name + " added\n")

    def reset_all(self):
        self.messages = []
        self.generated_text = ""
        self.content = []
        self.functions = []
        self.available_functions = {}
        self.token_count = 0
    
    def reset_messages(self):
        self.messages = []
        self.generated_text = ""
        self.content = []
        self.token_count = 0



















class EmbeddingsGenerator:
    def __init__(self, name):
        self.name = name
        self.dfs = {}


    def load_csv(self, csv_name):
        df = pd.read_csv(csv_name)
        csv_name = csv_name[:-4]
        self.dfs[csv_name] = df
        print("Dataframe created.")


    def load_csv(self, csv_name):
        df = pd.read_csv(csv_name)
        
        df['embedding'] = df['embedding'].apply(ast.literal_eval)  # convert string to list
        df['embedding'] = df['embedding'].apply(np.array)  # convert list to numpy array
        csv_name = csv_name[:-4]
        self.dfs[csv_name] = df
        print("Dataframe created.")


    def embedd_df(self, df_name):
        df = self.dfs[df_name]
        # check if embeddings rows contains no values
        if df["embedding"].isnull().values.any():
            df["embedding"] = df[df_name].apply(lambda x: get_embedding(x, engine=embedding_model))
            if isinstance(df["embedding"].iloc[0], np.ndarray):
                df["embedding"] = df["embedding"].apply(lambda x: x.tolist())  # convert numpy array to list
            print("Embeddings Generated, and csv amended.")
            
            df['embedding'] = df['embedding'].apply(ast.literal_eval)  # convert string to list
            df['embedding'] = df['embedding'].apply(np.array)  # convert list to numpy array

            df.to_csv(df_name + ".csv", index=False)
        else:
            print("Embeddings already loaded.")


    def semantic_search(self, column, query, n=1):
        df = self.dfs[column]
        query_embedding = get_embedding(query, engine=embedding_model)
        df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, query_embedding))
        results = df.sort_values("similarity", ascending=False, ignore_index=True).head(n)
        return results

    def additional_search(self, column, row, query, n=1):
        df = self.dfs[column]
        query_embedding = get_embedding(query, engine=embedding_model)

        result = self.dfs[column]["similarity"]

        



        

        

        




    

    


    def cull_by_max_tokens(self, column):
        encoding = tiktoken.get_encoding(embedding_encoding)
        df = self.dfs[column]
        df["n_tokens"] = df[column].apply(lambda x: len(encoding.encode(x)))
        df = df[df["n_tokens"] <= max_tokens]
        self.dfs[column] = df

def count_tokens(prompt, model):

    if model == "gpt-4":
        tokens = encoding4.encode(prompt)
        num_tokens = len(tokens)  
        return num_tokens

    elif model == "gpt-3.5-turbo":
        tokens = encoding35.encode(prompt)
        num_tokens = len(tokens)
        return num_tokens

    elif model == "text-embedding-ada-002":
        tokens = embedding_encoding.encode(prompt)
        num_tokens = len(tokens)
        return num_tokens


#This function takes an array of strings and joins them into a single string with each element separated by a newline character.
def join_array_into_string(array):
    result = '\n'.join(array)
    return result
