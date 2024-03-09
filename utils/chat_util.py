import tiktoken
import json
import os
from openai import OpenAI

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
tokenizer = tiktoken.get_encoding(encoding)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



def Chat(params):
    try:
        response = client.chat.completions.create(**params)
        response_message = response.choices[0].message 
        usage = response.usage

        if response_message.tool_calls:
            print("\n\nfunction call detected.\n\n")
            return (response_message.tool_calls, usage)
        
        else:
            response_text = response.choices[0].message.content
            print("\n\nGenerator call detected.\n\n")
            return (response_text, usage)
        
    except Exception as e:
        print(e)



def function_response(response, available_functions):

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


class Model_Widgets:
    def __init__(self):
        self.functions = []
        self.available_functions = {}


    def token_counter(self, model, input_messages, output_messages, functions=None):

        input_tokens = tokenizer.encode(str(input_messages))
        input_token_count += len(input_tokens)

        output_tokens = tokenizer.encode(str(output_messages))
        output_token_count += len(output_tokens)

        if functions is not None:
            for func in functions:
                input_func_tokens = tokenizer.encode(str(func))
                input_token_count += len(input_func_tokens)

        # todo: calculate input and output token costs, based on model.


    def add_available_function(self, name, function=None):
        self.available_functions[name] = function
        print(name + " added\n")


