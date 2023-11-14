tools = [
    {
        "type": "function",
        "function": {
            "name": "ner_function",
            "description": "Extracts named entities and their custom categories from the input text based on the analysis of customer support emails for a webstore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity": {
                                    "type": "string", 
                                    "description": "A Named entity extracted from the customer service email chain."
                                },
                                "category": {
                                    "type": "string", 
                                    "description": "Custom category of the named entity based on the analysis of the email chain."
                                }
                            }                            
                        }
                    }
                }
            },
            "required": ["entities"]
        }
    }
]

tools2 = [
    {
        "type": "function",
        "function": {
            "name": "INSERT_NAME_HERE",
            "description": "Generate categories and entities from a given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_file": {
                        "type": "object",
                        "properties": {
                            "Categories": {
                                "type": "object",
                                "description": "names of the categories",
                                "properties": {
                                    "Category": {
                                        "type": "object",
                                        "description": "individual category object with category name as the key",
                                        "properties": {
                                            "type": "string",
                                            "description": "item within the category with it's values"
                                        }
                                    }
                                }   
                            }                      
                        }
                    }
                }
            },
            "required": ["json_file"]
        }
    }
]