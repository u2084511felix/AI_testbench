import json


"""

Route agent-modules to create the AI pipelines.

"""

basic_chat_dict = open('./agent_modules/basic_chat/basic_chat.json').read()

basic_chat_dict = json.loads(basic_chat_dict)

basic_chat_dict["model"] = "gpt-4"

with open('./agent_modules/basic_chat/basic_chat.json', 'w') as f:
    json.dump(basic_chat_dict, f)

print(basic_chat_dict)