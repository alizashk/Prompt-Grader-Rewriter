# Prompt Grader & Rewriter — U1 to U5

## Included modules
- **U1:** Five-part prompt rubric and transparent keyword baseline.
- **U2:** Gemini LLM judge with few-shot examples and Gemini prompt rewriting.
- **U3:** DistilBERT prompt-quality classifier training script + CLIP text/image similarity.
- **U4:** Run original and rewritten text prompts; optional Gemini image generation.
- **U5:** Rule-based checks for potentially biased wording and personal-data patterns.

## 1. Setup (Windows / VS Code)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add your own Gemini API key:
```env
GEMINI_API_KEY=your_actual_key
GEMINI_MODEL=gemini-2.5-flash
```
Run:
```bash
streamlit run app.py
```

## 2. Enable U3 models
Install the extra ML packages (large download):
```bash
pip install -r requirements-ml.txt
```
Train the demonstration BERT classifier:
```bash
python scripts/train_bert.py
```
The training script uses a tiny synthetic teaching dataset to demonstrate the workflow. It is not a validated benchmark. For a credible project result, collect and manually label a larger, diverse prompt dataset, split by prompt (not duplicated rows), train/evaluate, and report actual metrics only.

CLIP downloads `openai/clip-vit-base-patch32` the first time it is used. Upload an image and the text prompt to calculate text-image similarity. CLIP is not a prompt-quality grader; it measures image/text alignment.

## 3. U4 image generation
Image generation depends on the model name and feature access available to your Google AI Studio API key. If the configured image model is unavailable, text execution still works. Update `GEMINI_IMAGE_MODEL` to a currently supported image-generation model for your account.

## 4. U5 limitations
The bias and personal-data checks are rule-based demonstrations. They are not complete privacy protection, do not send prompt text to a separate safety provider, and may miss sensitive content or flag harmless content. Do not use real personal data in tests. Human review is required.

## Security
Never commit `.env` or share your API key. If exposed, revoke it and create a new key.

## Honest project status
The app includes starter implementations for U1–U5, but U3 model training and U4 image generation require separate setup and may need a capable computer/API entitlement. Do not claim model accuracy or safety guarantees without running tests and collecting evidence.
