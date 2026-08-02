from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

# Import the HNIF framework from the hnif package
from hnif.adapter import run_hnif_analysis

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
app = FastAPI(title="Forensic AI Audit API", version="1.0")

# Enable CORS so your Next.js frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # In production, replace "*" with your Next.js URL (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the base classification model
model = None
tokenizer = None
device = None


# ==============================================================================
# 2. DATA MODELS (API Request/Response schemas)
# ==============================================================================
class AuditRequest(BaseModel):
    text: str


# ==============================================================================
# 3. SERVER LIFECYCLE (Load Base Model on Startup)
# ==============================================================================
@app.on_event("startup")
async def load_models():
    """
    Initializes and loads the base transformer model into memory on server startup
    to prevent repetitive I/O overhead during inference requests.
    """
    global model, tokenizer, device
    print("🚀 Starting Server: Loading Base Classifier into memory...")

    # Path to the fine-tuned detection weights (Updated for root execution)
    LOCAL_MODEL_DIR = "./hnif/models"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Hardware detected: {device.upper()}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        # output_attentions=True is required for the HNIF extraction layer
        model = AutoModelForSequenceClassification.from_pretrained(
            LOCAL_MODEL_DIR, output_attentions=True
        )
        model.to(device)
        model.eval()
        print(f"✅ Base Classifier loaded successfully on {device.upper()}!")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")


# ==============================================================================
# 4. INFERENCE PIPELINE
# ==============================================================================
def perform_classification(text: str):
    """
    Executes a forward pass on the base classification model.
    Returns: Tuple[str, float] representing (predicted_class_name, confidence_percentage)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Assuming Class 1 is "AI-Generated" and Class 0 is "Human-Written"
    ai_prob = probs[0][1].item()
    human_prob = probs[0][0].item()

    if ai_prob > human_prob:
        return "AI-Generated", round(ai_prob * 100, 2)
    else:
        return "Human-Written", round(human_prob * 100, 2)


# ==============================================================================
# 5. MAIN API ENDPOINT
# ==============================================================================
@app.post("/api/audit")
async def audit_text(request: AuditRequest):
    """
    Primary endpoint for processing forensic audit requests.
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    print(
        f"\n📩 Received new audit request. Sequence length: {len(request.text)} characters."
    )

    # STEP 1: Base Model Inference
    start_time = time.time()
    predicted_class, confidence = perform_classification(request.text)
    print(f"🧠 Classification Result: {predicted_class} ({confidence}%)")

    # STEP 2: Execute HNIF XAI Pipeline
    print("🔍 Running HNIF Intrinsic Extraction...")
    hnif_payload = run_hnif_analysis(
        model=model,
        tokenizer=tokenizer,
        text=request.text,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    total_time = time.time() - start_time
    print(f"✅ Audit pipeline completed in {total_time:.2f} seconds.")

    # STEP 3: Return standardized JSON response
    return hnif_payload
