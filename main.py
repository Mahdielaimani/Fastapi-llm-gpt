from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Define the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model
text_generator = pipeline("text-generation", model="maneln/fine-tuning-LLM-gpt")

# Define the request model
class TextGenerationRequest(BaseModel):
    prompt: str

# Define the response model
class TextGenerationResponse(BaseModel):
    response: str

# Define the endpoint for text generation
@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        result = text_generator(request.prompt, max_length=100)
        return TextGenerationResponse(response=result[0]['generated_text'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the text generation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
