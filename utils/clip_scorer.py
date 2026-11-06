"""U3 CLIP text-image similarity helper."""
def score_image_prompt(prompt, image):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies: pip install -r requirements-ml.txt") from exc
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity = outputs.logits_per_image.softmax(dim=1)[0, 0].item()
    return similarity
