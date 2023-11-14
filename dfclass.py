import sys
import tiktoken

import os
import pandas as pd
import numpy as np
import ast
import json
import csv
from scipy.spatial.distance import cosine
import ast
import regex as re
from pprint import pprint


from openai import OpenAI
import openai_utils

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)



# LLM models
gpt4 = "gpt-4"
gpt46 = "gpt-4-0613"

gpt35616k = "gpt-3.5-turbo-16k"
gpt411 = "gpt-4-1106-preview"
gpt3511 = "gpt-3.5-turbo-1106"

# embedding model parameters
embedding_model = "text-embedding-ada-002"

encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
tokenizer = tiktoken.get_encoding(encoding)

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


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model)['data'][0]['embedding']


def cosine_similarity(self, vec1, vec2):
    # Manual cosine similarity calculation
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
            
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # Handle division by zero
            
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def llmagent(messages, function_specifier="auto", functions=None, model=gpt3511, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0, max_tokens=0):
    
    try:
        params = {
            'model': model,
            'temperature': temperature,
            'messages': messages,
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
        }

        if max_tokens != 0:
            params['max_tokens'] = max_tokens
        
        if functions is not None:
            params['tools'] = functions
            params['tool_choice'] = {
                'type': 'function', 
                'function': {
                    'name': function_specifier
                }
            }
        
        response = client.chat.completions.create(**params)
        
        response_message = response.choices[0].message
        usage = response.usage

        total_tokens = usage.total_tokens
        print(total_tokens)



        if response_message.tool_calls:
            print("\n\nfunction call detected.\n\n")
            print(response_message)
            print()
            return (response_message.tool_calls, usage)
        
        else:
            response_text = response.choices[0].message.content
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
            print("tool call detected\n", tool_call, "\n")
            
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
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

        for msg in self.messages:
            tokens = tokenizer.encode(str(msg))
            self.token_count += len(tokens)

        if functions is not None:
            for func in functions:
                tokens = tokenizer.encode(str(func))
                self.token_count += len(tokens)

        print(f"\n\nToken count: {self.token_count}\n\n")
    

    def dynamic_model_change(self):
        if self.model == gpt3511 and self.token_count > 4000:
            print("\n\nToken count exceeded for gpt-3.5-turbo. Switching to gpt-3.5-turbo-16k\n\n")
            self.model = gpt35616k

    def system(self, msg):
        if self.messages != []:
            self.messages[0]({"role": "system", "content": msg})
        else:
            self.messages.append({"role": "system", "content": msg})


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




class Threads:
    def __init__(self, name):
        self.name = name
        self.threads = []
        self.files = []
        self.file_ids = []
        self.assistant_ids = []
        self.tools = []
        self.thread_ids = []

    def update_tools(self, tool):
        self.tools.append(tool)
        # eg. {"type": "retrieval"}


    def add_file(self, file):
        file = client.files.create(
            file=open(file, 'rb'),
            purpose='assistants'
        )

        file_id = file.id
        print("uploaded file. file id: ", file_id)
        self.file_ids.append(file_id)
    
    
    def create_assistant(self, model, instructions):

        assistant = client.beta.assistants.create(
            instructions=instructions,
            model=model,
            tools=self.tools,
            file_ids=self.file_ids,
        )

        assistant_id = assistant.id
        print("assistant created. assistant id: ", assistant_id)
        self.assistant_ids.append(assistant_id)
        return assistant_id
    
    # optional
    def update_assistant(self, assistant_id, instructions, model):
        assistant = client.beta.assistants.update(
            assistant_id,
            instructions=instructions,
            model = model,
            tools=self.tools,
            file_ids=self.file_ids,
        )
        assistant_id = assistant.id
        print("assistant updated. assistant id: ", assistant_id)
        return assistant_id

    # optional
    def retrieve_assistant(self, assistant_id):
        assistant = client.beta.assistants.retrieve(assistant_id)
        print("retreived assistant: ", assistant_id)
        return assistant

    def create_thread(self):
        run = client.beta.threads.create()
        thread_id = run.id
        self.thread_ids.append(thread_id)
        print("thread created. thread id: ", thread_id)
        return thread_id

    def add_message_to_thread(self, thread_id, content):
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        print("message added to thread: ", thread_id)
        return message

# =======================================================================================
# =======================================================================================

    def run_assistant_with_message(self, thread_id, assistant_id, instructions):
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions=instructions
        )

        run_id = run.id
    
        while run.status not in ["completed", "failed"]:
            run = client.beta.threads.runs.retrieve(
                thread_id= thread_id,
                run_id= run_id
            )
            print("run status: ", run.status)


        print("run complete. run id: ", run_id)

    
    def retrieve_run_responses(self, thread_id):
        messages = client.beta.threads.messages.list(
            thread_id=thread_id,
        )

        last_message = ""
        for each in messages:
            #print(each.role, ": ", each.content[0].text.value)
            last_message = each.content[0].text.value
            print(last_message)

        return last_message

    def delete_assistant(self, assistant_id):
        response = client.beta.assistants.delete(assistant_id)
        print(response)
        self.assistant_ids.remove(assistant_id)
        return response












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
        df = self.dfs[column]
        df["n_tokens"] = df[column].apply(lambda x: len(tokenizer.encode(x)))
        df = df[df["n_tokens"] <= max_tokens]
        self.dfs[column] = df



def count_tokens(prompt):
    tokens = tokenizer.encode(prompt)
    num_tokens = len(tokens)  
    return num_tokens



#This function takes an array of strings and joins them into a single string with each element separated by a newline character.
def join_array_into_string(array):
    result = '\n'.join(array)
    return result
