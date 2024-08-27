import sys
import openai
import json

file_path =  "./data/stepgame_valid/valid_demo.json"
if __name__ == "__main__":

    with open(file_path) as f:
        dataset = json.load(f)    
    
    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "none"
    for i in range(0, len(dataset)):
        for chunk in openai.ChatCompletion.create(
            model="main",
            messages=[
                {"role": "user", "content": dataset[i]["instruction"]}
            ],
            temperature=0.95,
            stream=True
        ):
            if hasattr(chunk.choices[0].delta, "content"):
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")        
