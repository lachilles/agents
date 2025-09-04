from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import glob
from pathlib import Path


load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Lianne Achilles"
        self.resources = {}
        self._load_all_resources()

    def _load_all_resources(self):
        """Dynamically load all resources from the me/ directory"""
        me_dir = Path("me")
        if not me_dir.exists():
            print("Warning: me/ directory not found")
            return
        
        # Get all files in the me/ directory
        for file_path in me_dir.iterdir():
            if file_path.is_file():
                try:
                    content = self._read_file_content(file_path)
                    if content:
                        # Use filename without extension as key
                        key = file_path.stem
                        self.resources[key] = content
                        print(f"Loaded resource: {file_path.name}")
                except Exception as e:
                    print(f"Error reading {file_path.name}: {e}")

    def _read_file_content(self, file_path):
        """Read content from various file types"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._read_pdf(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._read_text_file(file_path)
        else:
            # Try to read as text file for other extensions
            try:
                return self._read_text_file(file_path)
            except:
                print(f"Unsupported file type: {file_extension}")
                return None

    def _read_pdf(self, file_path):
        """Read content from PDF files"""
        try:
            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
            return content.strip()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return None

    def _read_text_file(self, file_path):
        """Read content from text files"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return None


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given various resources about {self.name} which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        # Dynamically add all loaded resources to the system prompt
        if self.resources:
            system_prompt += "\n\n## Available Resources:\n"
            for resource_name, content in self.resources.items():
                system_prompt += f"\n### {resource_name.title()}:\n{content}\n"
        else:
            system_prompt += "\n\n## Note: No resources were loaded from the me/ directory."
        
        system_prompt += f"\n\nWith this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch(share=True)
    