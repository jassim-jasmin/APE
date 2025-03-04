#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Mohammed Jassim at 04/03/25
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, LogitsProcessorList, MinLengthLogitsProcessor
import torch

# Initialize FastAPI app
app = FastAPI()

# Check if MPS is available (for Apple Silicon)
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'

# Load Hugging Face model
model_name = "microsoft/phi-2"
generator = pipeline(
    "text-generation",
    model=model_name,
    device=0 if device == "mps" else -1,  # Use MPS if available, else CPU
    torch_dtype=torch.float16 if device == "mps" else torch.float32
)

# Ensure pad token is set (to prevent warnings)
generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id


# Define request body structure
class PromptRequest(BaseModel):
    user_input: str


# Function to generate multiple prompt variations
def generate_prompt_variations(user_input):
    base_prompts = [
        f"{user_input} (explain in simple terms)",
        f"Provide a concise response for: {user_input}",
        f"Generate a step-by-step answer to: {user_input}",
        f"Summarize the following in 3 bullet points: {user_input}",
        f"Answer in 2 sentences: {user_input}"
    ]
    return base_prompts


# Function to evaluate and select the best prompt
def get_best_prompt(prompt_variations):
    best_prompt = None
    best_score = float("-inf")

    for prompt in prompt_variations:
        # Move inputs to MPS explicitly
        inputs = generator.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        response = generator(
            prompt,
            max_length=50,
            do_sample=True,
            temperature=0.7,
            truncation=True,  # Fix truncation warning
            logits_processor=LogitsProcessorList([
                MinLengthLogitsProcessor(
                    5, eos_token_id=generator.tokenizer.eos_token_id
                )
            ])
        )

        generated_text = response[0]["generated_text"]
        score = len(generated_text)

        if score > best_score:
            best_score = score
            best_prompt = prompt

    return best_prompt


# API Endpoint to generate optimized prompts
@app.post("/generate_prompt")
async def optimize_prompt(request: PromptRequest):
    try:
        print('Processing request...')
        prompt_variations = generate_prompt_variations(request.user_input)
        best_prompt = get_best_prompt(prompt_variations)

        response = {
            "original_input": request.user_input,
            "optimized_prompt": best_prompt
        }

        print(response)
        return response

    except Exception as e:
        print(f"Exception: {e}")
        raise e


# Run with: uvicorn script_name:app --reload
