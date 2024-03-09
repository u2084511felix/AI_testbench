import json
from chat_util import Chat


"""
Allows for cross-module I/O routing by allowing the user to specify the module to which the message is being sent.

"""




def input(messages_path, role, message_content):

    messages = json.loads(open(messages_path).read())
    messages.append({"role": role, "content": message_content})

    with open(messages_path, 'w') as f:
        json.dump(messages, f)

    return messages


def output(messages_path, module):

    messages = json.loads(open(messages_path).read())
    module["messages"] = messages
    
    response, usage = Chat(module)

    return response, usage