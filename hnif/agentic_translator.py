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
                "word": item["display_token"],
                "impact": round(item["normalized_score"], 2),
            }
            for item in sorted_words[:5]
        ]

        # 2. Construct the Strict System Prompt
        system_prompt = (
            "You are an expert Digital Forensic AI analyzing text for an Academic Integrity Board. "
            "Translate the mathematical tensor data into a professional, objective 3-sentence audit report. "
            "Do not hallucinate. Only reference the exact words and scores provided in the payload. "
            "Explain exactly why the model flagged the text based on these specific causal words."
        )

        user_prompt = (
            f"Classification: {classification} (Confidence: {confidence:.2f}%)\n"
            f"Top Causal Triggers Isolated by HNIF:\n{json.dumps(top_words, indent=2)}\n\n"
            "Generate the 3-sentence professional summary now:"
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
