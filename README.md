# APE (**Automated Prompt Engineering**)
This project provides an API for generating optimized text prompts using a Hugging Face `transformers` model. It uses FastAPI for handling API requests and the `microsoft/phi-2` model for text generation.

## Features
- Accepts user input and generates multiple variations of prompts.
- Selects the best prompt based on text length.
- Utilizes Apple's MPS backend if available (for Apple Silicon users).
- Runs on CPU if MPS is unavailable.

## Requirements
- `python 3.11`
- `fastapi`
- `pytorch`
- Open source model `mistralai/Mistral-7B-Instruct-v0.1`

## How This Works
	1.	Loads an open-source model like Mistral-7B (or Llama-2).
	2.	Generates multiple prompt variations.
	3.	Scores the best prompt using AI-generated text length.
	4.	Returns the optimized prompt for better AI interaction.

## How to Use the API

### Install Dependencies
```bash
pip install fastapi uvicorn torch transformers pydantic
```

## Running the API

To start the FastAPI server, run:
```bash
uvicorn app:app --reload
```

or

`python -m uvicorn app:app`


The API will be available at:
```
http://127.0.0.1:8000
```


## API Endpoints

### Generate Optimized Prompt
#### **POST /generate_prompt**

**Request Body:**
```json
{
  "user_input": "Explain quantum computing"
}
```

**Response:**
```json
{
  "original_input": "Explain quantum computing",
  "optimized_prompt": "Provide a concise response for: Explain quantum computing"
}
```

## Configuration

- The model used is `microsoft/phi-2`, but you can change it in `model_name`.
- The device is set to use Apple MPS if available, otherwise it defaults to CPU.

## Troubleshooting

### Common Errors
1. **"MPS not available" error**
   - Ensure you have a Mac with an M1/M2/M3 chip.
   - Update `torch` with:
     ```bash
     pip install --upgrade torch
     ```

2. **"Model not found" error**
   - Download the model manually or ensure you have an internet connection.

## License
This project is open-source and licensed under the MIT License.


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


### Other model options

      Model	Size	 Best         For
      Mistral-7B	 7B           General-purpose chat
      Llama-2-7B	 7B           Creative text generation
      Falcon-7B      7B           Open-source alternatives to GPT

Change the variable below in code

      model_name = "microsoft/phi-2"

## Enhancements You Can Add

      ✅ Fine-tune prompt selection using embeddings (e.g., FAISS, ChromaDB)
      ✅ Deploy the API with a GPU for faster inference
      ✅ Use reinforcement learning (RLHF) to improve prompt ranking






