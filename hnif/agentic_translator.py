import torch
import json
from typing import List, Dict, Any


class AgenticTranslator:
    """
    Translates raw tensor mathematical data into a human-readable forensic audit report
    using a locally hosted, quantized Large Language Model.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print("Initializing Agentic Translator (Local LLM)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            # 4-bit quantization compresses the model so it fits on standard GPUs alongside DeBERTa
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config if self.device == "cuda" else None,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.is_loaded = True
            print("Agentic Translator loaded successfully.")

        except Exception as e:
            print(
                f"⚠️ Warning: Failed to load local LLM. Ensure transformers & bitsandbytes are installed. Error: {e}"
            )
            self.is_loaded = False

    def generate_report(
        self, heatmap_data: List[Dict[str, Any]], classification: str, confidence: float
    ) -> str:
        """
        Extracts top causal words and prompts the LLM to generate the report.
        """
        if not self.is_loaded:
            return "Agentic Translator offline: Hardware or dependency limitation."

        # 1. Isolate the Top 5 Semantic Triggers (Ignoring Noise)
        valid_words = [item for item in heatmap_data if not item["is_noise"]]
        # Sort by raw score descending
        sorted_words = sorted(valid_words, key=lambda x: x["raw_score"], reverse=True)
        top_words = [
            {
                # Extra safety cleanup to remove any lingering DeBERTa symbols
                "word": item["display_token"]
                .replace(" ", "")
                .replace("\u2581", "")
                .strip(),
                "impact": round(item["normalized_score"], 2),
            }
            for item in sorted_words[:5]
        ]

        # 2. Construct the Strict System Prompt (Format Enforcement)
        system_prompt = (
            "You are an objective Digital Forensic AI API. Your ONLY job is to translate a JSON "
            "array of mathematical attention scores into a concise, strictly formatted 3-sentence summary.\n"
            "CRITICAL: DO NOT hallucinate context, intent, or real-world applications (e.g., NEVER mention 'academic integrity', 'research', or 'plagiarism').\n"
            "YOU MUST FOLLOW THIS EXACT FORMAT:\n"
            "Sentence 1: State the model's classification and the confidence percentage.\n"
            "Sentence 2: List the highest impact words and state that they were the primary statistical drivers.\n"
            "Sentence 3: Conclude objectively that these specific causal triggers caused the network's decision, without adding any outside context."
        )

        user_prompt = (
            f"Model Classification: {classification} (Confidence: {confidence:.2f}%)\n"
            f"Highest Impact Tokens Isolated by HNIF Attention Tensors:\n{json.dumps(top_words, indent=2)}\n\n"
            "Generate the 3-sentence analytical summary now following the exact format requested:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        # Generate deterministically (low temperature)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.2,  # Low temperature forces analytical, non-creative tone
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Clean and extract just the generated response
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response.strip()
