import os
import re
import json
import sys
import subprocess
import time
import datetime
import random
import csv
from ai_functions import tools, tools2
from dfclass import *

def parseEmailTranscript(transcript):

    transcript = open(transcript, 'r').read()

    emailLines = transcript.split('\n')
    emails = {}
    currentSubject = ''
    currentEmail = {}
    currentBody = ''
    isBody = False

    for line in emailLines:
        if (line.startswith('Subject:')):
            if (currentSubject != '' and currentEmail):
                currentEmail['body'] = currentBody.strip()
                if currentSubject not in emails:
                    emails[currentSubject] = []
                emails[currentSubject].append(currentEmail)
                
            currentSubject = line.replace('Subject:', '').replace('Re:', '').strip()
            currentEmail = {}
            currentBody = ''
            isBody = False
        elif (line.startswith('From:')):
            currentEmail['from'] = line.replace('From:', '').strip()
        elif (line.startswith('To:')):
            currentEmail['to'] = line.replace('To:', '').strip()
        elif (line.startswith('Date:')):
            isBody = True
        elif (line.startswith('Summary:')):
            currentEmail['summary'] = line.replace('Summary:', '').strip()
            if currentSubject != '' and currentEmail:
                currentEmail['body'] = currentBody.strip()
                if currentSubject not in emails:
                    emails[currentSubject] = []
                emails[currentSubject].append(currentEmail)
            currentSubject = ''
            currentEmail = {}
            currentBody = ''
            isBody = False
        elif (isBody):
            currentBody += line + '\n'

    return emails


def Process_chunks(emails):
    email_json_array ={}
    # create a string from each email
    for _subject in emails:
        subject_body = emails[_subject]
        subject_name = f"Subject: {_subject}"

        for email in subject_body:
    
            email_string = ''

            if ('summary' in email):
                continue

            email_string += f"{subject_name}. "
            email_string += "From: "
            email_string += email['from']
            email_string += ". "
            email_string += "To: "
            email_string += email['to']
            email_string += ". "
            email_string += "Body: "
            email_string += email['body']
            email_string += ". "

            key = _subject + email['from']

            email_json_array[key] = email_string

    return email_json_array


def ner_function(entities):
    print("Extracting NER from text")

    ner = {}

    for entity in entities:
        if entity['category'] not in ner:
            ner[entity['category']] = []
        ner[entity['category']].append(entity['entity'])


    if os.path.isfile('ner.json'):
        with open('ner.json', 'r') as f:
            data = json.load(f)
            # update data with new ner
            for category in ner:
                if category not in data:
                    print(f"New category found: {category}")
                    data[category] = []
                for entity in ner[category]:
                    print(f"New entity found: {entity}")
                    if entity not in data[category]:
                        data[category].append(entity)

        with open('ner.json', 'w') as f:
            json.dump(data, f, indent=4)

    else:
        with open('ner.json', 'w') as f:
            for category in ner:
                print(f"New category found: {category}")
                print(f"New entities found: {ner[category]}")
            json.dump(ner, f, indent=4)

    print("NER extraction complete, saved to ner.json")
    return ("NER extraction complete, saved to ner.json")







if __name__ == '__main__':

    embedd = EmbeddingsGenerator("emails")


# ====================================================================================================
# ====================================================================================================

    # emails = parseEmailTranscript('transcript.txt')
        
    # _subject = {"subject" : [], "embedding" : []}
    # _body = {"body" : [], "embedding" : []}
    # _summary = {"summary" : [], "embedding" : []}


    # for subject in list(emails.keys()):
    #     # stringify subject for json dump using regex
    #     sanitized_subject = re.sub(r'[^a-zA-Z0-9\s]', '', subject)
    #     # remove newline and tab characters
    #     sanitized_subject = re.sub(r'[\n\r]', ' ', sanitized_subject)

    #     _subject["subject"].append(sanitized_subject)
    #     for email in emails[subject]:
    #         if (email.get('body') is not None):
    #             # stringify body for json dump using regex
    #             email['body'] = re.sub(r'[^a-zA-Z0-9\s]', '', email['body'])
    #             # remove newline and tab characters
    #             email['body'] = re.sub(r'[\n\r]', ' ', email['body'])
    #             _body["body"].append(email['body'])
    #         if (email.get('summary') is not None):
    #             # stringify summary for json dump using regex
    #             email['summary'] = re.sub(r'[^a-zA-Z0-9\s]', '', email['summary'])
    #             # remove newline and tab characters
    #             email['summary'] = re.sub(r'[\n\r]', ' ', email['summary'])
    #             _summary["summary"].append(email['summary'])
    

    # # write to json file
    # with open('emails.json', 'w') as outfile:
    #     json.dump(emails, outfile, indent=4)

    # with open('subject.csv', 'w') as outfile:
    #     writer = csv.writer(outfile)
    #     writer.writerow(_subject.keys())
    #     for val in _subject["subject"]:
    #         writer.writerow([val, ""])
    # with open('body.csv', 'w') as outfile:
    #     writer = csv.writer(outfile)
    #     writer.writerow(_body.keys())
    #     for val in _body["body"]:
    #         writer.writerow([val, ""])
    # with open('summary.csv', 'w') as outfile:
    #     writer = csv.writer(outfile)
    #     writer.writerow(_summary.keys())
    #     for val in _summary["summary"]:
    #         writer.writerow([val, ""])


# ====================================================================================================
# ====================================================================================================


    # embedd.load_csv("subject.csv")
    # embedd.load_csv("body.csv")
    # embedd.load_csv("summary.csv")

    # embedd.embedd_df("subject")
    # embedd.embedd_df("body")
    # embedd.embedd_df("summary")


    # results = embedd.semantic_search("subject", "What are your return policies?", n=1)

    # load json emails


# ====================================================================================================
# ====================================================================================================


    #gen = Generator("emails")
    #gen.add_available_function("ner_function", ner_function)

    #emails = json.load(open('emails.json', 'r'))

    # for subject in list(emails.keys()):
    #     for email in emails[subject]:
    #         if (email.get('body') is not None):
    #             gen.reset_messages()
    #             # stringify body for json dump using regex
    #             email['body'] = re.sub(r'[^a-zA-Z0-9\s]', '', email['body'])
    #             email['body'] = re.sub(r'[\n\r]', ' ', email['body'])
    #             text = email['body']
    #             gen.run(msg=text, function_specifier="ner_function", model="gpt-4-1106-preview", temperature=0, max_tokens=3988, functions=tools)

    

    transcript = open('transcript.txt', 'r').read()

    # count to ten for loop


    prompt = "For the given text please extract key data variables and their values. This is for amalgamating company data from a given dataset from a companies files. It could be emails, company documents, legal documents, financial documents, etc. The extracted data needs to be accessible in a chatbot which will compare the user query to the extracted data and return the most relevant data. Omit any personally identifiable information. Extract only key data that can be provided in different contexts and is not specific to a single email or document."

    #gen.system(prompt)



    #gen.run(msg=transcript, function_specifier="ner_function", model="gpt-4-1106-preview", temperature=0, max_tokens=4000, functions=tools)
    # gen.run(msg=transcript, model="gpt-4-1106-preview", temperature=0, max_tokens=4000)

    # print(gen.generated_text)


# ====================================================================================================
# ====================================================================================================

    thread = Threads("test")

    thread.update_tools({"type": "retrieval"})

    thread.add_file("transcript.txt")

    instructions = "retrieve with the most relevant information from given files."

    assistant_id = thread.create_assistant("gpt-4-1106-preview", instructions)

    thread_id = thread.create_thread()



    def thread_convo(query):
        thread.add_message_to_thread(thread_id, query)

        chatbot_instructions = "please reply to the user from the perspective of the customer service chatbot for the company using the given company data"

        thread.run_assistant_with_message(thread_id, assistant_id, chatbot_instructions)

        thread.retrieve_run_responses(thread_id)
    


    while True:
        query = input("Enter a query: ")
        thread_convo(query)

    




    # for r in results:
    #     print(r)
    #     print("")