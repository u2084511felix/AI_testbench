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
gpt356 = "gpt-3.5-turbo-0613"
gpt35616k = "gpt-3.5-turbo-16k-0613"
gpt4 = "gpt-4"
gpt46 = "gpt-4-0613"

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



def llmagent(messages, function_specifier="auto", functions=None, model=gpt356, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, max_tokens=0):
    
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
            params["functions"] = functions
            params["function_call"] = {"name": function_specifier}
        
        response = openai.ChatCompletion.create(**params)

        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            print("\n\nfunction call detected.\n\n")
            print(response_message)
            print()
            return response_message["function_call"]
        
        else:
            response_text = response["choices"][0]["message"]["content"]
            print("\n\nGenerator call detected.\n\n")
            print("\n\n", response_text, "\n\n")
            return response_text
        
    except Exception as e:
        print(e)


def function_response(response, available_functions):
    """
    Gets the LLM generated function_call JSON (response).
    Gets the function object by key name from generators available_functions dictionary.
    Gets the arguments from the function_call JSON.
    Calls the function with the arguments.
    
    Return value of function passed back to the generator class.
    (this should always be a string)



    """

    try:
        # TODO validate JSON response
        
        function_name = response["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response["arguments"])
        
        print(str(function_args))
        print()

        if function_args == {}:
            return_value = function_to_call()
            print(return_value)
            return return_value
        else:
            return_value = function_to_call(**function_args)
            print(str(return_value))
            print()
            return return_value

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
        self.model = ""

    def token_counter(self, msg):
        tokens = tokenizer.encode(str(msg))
        self.token_count += len(tokens)
        print(f"\n\nToken count: {self.token_count}\n\n")
    

    def dynamic_model_change(self, msg):
        tokens = tokenizer.encode(msg)
        
        if self.model == gpt356 and self.token_count > (4000 + len(tokens)):
            print("\n\nToken count exceeded for gpt356. Switching to gpt35616k\n\n")
            self.model = gpt35616k
            
        if self.model == gpt46 and self.token_count > (8000 + len(tokens)):
            print("\n\nToken count exceeded for gpt46. Switching to gpt35616k\n\n")
            self.model = gpt35616k



    def system(self, msg):
        self.messages[0] = {"role": "system", "content": msg}
        self.token_counter(msg)



    def run(self, msg="", function_specifier="auto", model=gpt356, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, max_tokens=None, functions=None, extended_response=0, chatty=False):
        """
        The extended response needs to be manually calculated.
        This is determined by the number of chained function calls you expect the llm to make.
        (not an exact science, but an expedient solution)
        
        """
        self.generated_text = ""


        self.model = model

        if msg != "":
            self.messages.append({"role": "user", "content": msg})
            self.token_counter(msg)
            self.dynamic_model_change(msg)
        

        response = llmagent(self.messages, function_specifier=function_specifier, model=self.model, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, max_tokens=max_tokens, functions=functions)

        if functions is not None:

            for func in functions:
                self.dynamic_model_change(str(func))


            if extended_response == 0:
                print("\nextended response not detected\n")
                print(response["name"] + " detected\n")

                returned_function = function_response(response, self.available_functions)
                self.token_counter(str(returned_function))
                self.dynamic_model_change(str(returned_function))
                self.content.append("Function call: " + response["name"] + " = " + str(returned_function))
                self.generated_text = ("Function call: " + response["name"] + " = " + str(returned_function))
        
            if extended_response == 1:

                print("\n1 extended response detected\n")

                returned_function = function_response(response, self.available_functions)

                self.content.append("Function call: " + response["name"] + " = " + str(returned_function))

                self.messages.append({"role": "function", "name": response["name"], "content": str(returned_function)})
                self.token_counter(str(returned_function))
                self.dynamic_model_change(str(returned_function))

                self.generated_text = llmagent(self.messages, model=gpt356)

                self.content.append(self.generated_text)

            if extended_response > 1:

                iterator = 1
                
                for i in range(1, extended_response):
                    print(f"\nExtended response {i} of {extended_response}\n")
                    print(response["name"])
                    print()

                    returned_function = function_response(response, self.available_functions)
                
                    self.content.append("Function call: " + response["name"] + " = " + str(returned_function))

                    self.messages.append({"role": "function", "name": response["name"], "content": str(returned_function)})
                    
                    
                    
                    self.token_counter(str(returned_function))
                    self.dynamic_model_change(str(returned_function))
                    if chatty:
                        self.generated_text = llmagent(self.messages, model=gpt356)

                        self.content.append(self.generated_text)

                    response = llmagent(self.messages,function_specifier=function_specifier, model=self.model, functions=functions, max_tokens=max_tokens)
                    iterator += 1
                    
                    if iterator == extended_response:
                        print(f"\nExtended response {iterator} of {extended_response}\n")
                        print(response["name"])
                        print()

                        returned_function = function_response(response, self.available_functions)
                    
                        self.content.append("Function call: " + response["name"] + " = " + str(returned_function))

                        self.messages.append({"role": "function", "name": response["name"], "content": str(returned_function)})
                        self.token_counter(str(returned_function))
                        self.dynamic_model_change(str(returned_function))
                        self.messages.append({"role": "user", "content": "Summarise the results of the tasks."})
                        self.token_counter("Summarise the results of the tasks.")
                        self.dynamic_model_change("Summarise the results of the tasks.")
                        self.generated_text = llmagent(self.messages, model=gpt356)
                        self.messages.append({"role": "assistant", "content": self.generated_text})
                        self.token_counter(self.generated_text)
                        self.dynamic_model_change(self.generated_text)
                        self.content.append(self.generated_text)

        else:
            self.content.append(response)
            self.generated_text = response
            self.token_counter(self.generated_text)


    def aggregate(self, contentobj):
        contentobj.content[self.name]["content"] = self.content


    def add_available_function(self, name, function=None):
        self.available_functions[name] = function
        print(name + " added\n")

    def reset(self):
        self.messages = []
        self.generated_text = ""
        self.content = []
        self.functions = []
        self.available_functions = {}
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
