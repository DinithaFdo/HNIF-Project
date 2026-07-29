import time
from typing import Dict, Any
from .core_extractor import TensorExtractor
from .linguistic_filter import LinguisticNoiseFilter
from .agentic_translator import AgenticTranslator

# Initialize the filters and models once globally so they don't reload on every API request
_global_filter = LinguisticNoiseFilter()
_global_translator = AgenticTranslator()


def run_hnif_analysis(
    model: Any,
    tokenizer: Any,
    text: str,
    predicted_class: str = "AI-Generated",
    confidence: float = 99.0,
) -> Dict[str, Any]:
    """
    The main entry point for the HNIF middleware.

    Args:
        model (Any): A loaded HuggingFace Transformer classification model.
        tokenizer (Any): The corresponding tokenizer.
        text (str): The text to be audited.

    Returns:
        Dict[str, Any]: A structured payload containing metrics,
                        extracted data, and eventually the UI/Reports.
    """
    start_time = time.time()

    # 1. Initialize the Extractor
    extractor = TensorExtractor(model, tokenizer)

    # 2. Run the extraction (Currently only Attention, Gradients will be added later)
    attention_scores, raw_tokens = extractor.extract_attention(text)

    # 3. Apply Linguistic Filtration and Normalization for Frontend
    heatmap_data = _global_filter.process_and_normalize(raw_tokens, attention_scores)

    # 4. Generate Natural Language Report using the Agentic Translator
    report = _global_translator.generate_report(
        heatmap_data, predicted_class, confidence
    )

    execution_time = time.time() - start_time

    # 5. Construct the Strict Next.js API Payload
    result_payload = {
        "status": "success",
        "metadata": {
            "execution_time_seconds": round(execution_time, 4),
            "model_analyzed": (
                model.config.name_or_path
                if hasattr(model.config, "name_or_path")
                else "Unknown Transformer"
            ),
            "total_tokens": len(raw_tokens),
            "predicted_class": predicted_class,
            "confidence": confidence,
        },
        "heatmap_data": heatmap_data,
        "agentic_report": report,
    }

    return result_payload
