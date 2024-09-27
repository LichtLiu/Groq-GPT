from groq import Groq
import json
import gradio as gr
import torch
from datasets import load_dataset

# For textToSpeech
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf


# API key
GROQ_API_KEY = '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'
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-70b-versatile"




def calculate(expression):
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except:
        return json.dumps({"error": "Invalid expression"})
    

# text to speech block
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def textToSpeech(expression):
    inputs = processor(text=expression, return_tensors = "pt")
    speech = model.generate_speech(inputs["inputs_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("speech.wav", speech.numpy(), samplerate=16000)

def chatbot(user_prompt, history=[]):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
        ,
        {
            "type": "function",
            "function": {
                "name": "text-to-speech",
                "description": "text to speech",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "text to speech",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
        
    ]
    
    messages = history + [
    {
        "role": "system",
        "content": "You are a assistant. You can Use the calculate function to perform mathematical operations and provide the results and Answer question that is not mathematical question and text to audio."
    },
    {
        "role": "user",
        "content": user_prompt,
    }
    ] # Start with the system message and append the history

    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    response_message = response.choices[0].message

    tool_calls = response_message.tool_calls
    if tool_calls:
        available_functions = {
            "calculate": calculate,
            "text-to-speech": textToSpeech,
        }
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                expression=function_args.get("expression")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return second_response.choices[0].message.content, messages[1:]  # Return updated history
    else:
        return response.choices[0].message.content, messages[1:]


# Create Gradio interface with examples
examples = [
    "25 * 4 + 11",
    "100 / 5 - 7",
    "(3 + 5) * (10 - 2)",
    "2 ** 8",
    "50 % 3",
    "generate this text to audio: I like to sing, and calculate 25 * 4 + 11"
]

def gradio_chatbot(user_prompt, history=[]):
    response, new_history = chatbot(user_prompt, history)
    return response, new_history

demo = gr.Interface(
    fn=gradio_chatbot, 
    inputs=["text", "state"], 
    outputs=["text", "state"], 
    title="Groq Chatbot",
    examples=examples
)

# Launch the Gradio app
demo.launch(debug=True)

