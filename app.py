import streamlit as st
from PIL import Image
from utils.prompt_grader import grade_prompt, get_feedback
from utils.llm_judge import grade_prompt_with_llm, rewrite_prompt, run_text_prompt, generate_image
from utils.safety_checks import run_safety_checks
from utils.clip_scorer import score_image_prompt
from utils.bert_classifier import predict_prompt_quality, bert_model_available

st.set_page_config(page_title="Prompt Grader & Rewriter", page_icon="🤖", layout="wide")
st.title("🤖 Prompt Grader & Rewriter")
st.write("A mini-project for evaluating, improving, executing and checking prompts.")


with st.sidebar:
    st.header("Project modules")
    st.markdown("- U1: Five-part rubric")
    st.markdown("- U2: Gemini few-shot LLM judge")
    st.markdown("- U3: BERT classifier + CLIP image alignment")
    st.markdown("- U4: Run text and image prompts")
    st.markdown("- U5: Bias and personal-data checks")

tabs = st.tabs(["U1 · Baseline", "U2 · LLM Judge & Rewrite", "U3 · BERT + CLIP", "U4 · Run Prompts", "U5 · Safety Checks"])

with tabs[0]:
    st.header("U1 — Define the rubric")
    st.write("Each category is scored from 0 to 20, for a total of 100.")
    prompt = st.text_area("Enter a prompt", key="u1_prompt", height=130,
                          placeholder="Act as a Python teacher. Explain loops for beginners using bullet points in under 200 words.")
    if st.button("Grade with keyword baseline", key="u1_grade"):
        if prompt.strip():
            scores, total = grade_prompt(prompt)
            st.metric("Baseline score", f"{total}/100")
            cols = st.columns(5)
            for col, (name, score) in zip(cols, scores.items()):
                col.metric(name, f"{score}/20")
            for suggestion in get_feedback(scores):
                st.warning(suggestion)
            st.caption("This is a transparent keyword-based baseline, not a semantic AI judgment.")
        else:
            st.warning("Enter a prompt first.")

with tabs[1]:
    st.header("U2 — Few-shot LLM Judge")
    st.write("Gemini evaluates the five rubric dimensions using two scored examples.")
    p2 = st.text_area("Prompt to evaluate", key="u2_prompt", height=130)
    if st.button("Evaluate with Gemini", type="primary", key="u2_eval"):
        if not p2.strip():
            st.warning("Enter a prompt first.")
        else:
            try:
                with st.spinner("Gemini is evaluating..."):
                    result = grade_prompt_with_llm(p2)
                st.session_state["evaluated_prompt"] = p2
                st.session_state["evaluation_result"] = result
                st.metric("LLM Judge score", f"{result['total_score']}/100")
                cols = st.columns(5)
                for col, (name, score) in zip(cols, result["scores"].items()):
                    col.metric(name, f"{score}/20")
                st.subheader("Suggestions")
                for suggestion in result.get("feedback", []):
                    st.info(suggestion)
                st.write(result.get("overall_feedback", ""))
            except Exception as e:
                st.error(f"LLM evaluation failed: {e}")
    st.divider()
    st.subheader("Prompt Rewriter")
    rewrite_source = st.text_area("Prompt to rewrite", key="rewrite_source",
                                  value=st.session_state.get("evaluated_prompt", ""))
    if st.button("Rewrite prompt", key="rewrite_btn"):
        if not rewrite_source.strip():
            st.warning("Enter a prompt first.")
        else:
            try:
                with st.spinner("Rewriting..."):
                    improved = rewrite_prompt(rewrite_source)
                st.session_state["rewritten_prompt"] = improved
                st.success("Rewritten prompt")
                st.code(improved)
            except Exception as e:
                st.error(f"Rewrite failed: {e}")

with tabs[2]:
    st.header("U3 — BERT classifier and CLIP score")
    st.subheader("BERT prompt-quality classifier")
    st.write("The classifier predicts one of three labels: Low, Medium or High. Train it first using the provided script.")
    st.caption("Model available: " + ("Yes" if bert_model_available() else "No — train the model first."))
    bert_text = st.text_area("Prompt for BERT classification", key="bert_prompt")
    if st.button("Predict prompt quality", key="bert_predict"):
        if not bert_text.strip():
            st.warning("Enter a prompt.")
        else:
            try:
                label, confidence = predict_prompt_quality(bert_text)
                st.metric("Predicted quality", label)
                st.write(f"Model confidence: {confidence:.1%}")
            except Exception as e:
                st.error(str(e))
    st.divider()
    st.subheader("CLIP image-prompt alignment")
    st.write("Upload an image and enter the prompt that was used to create it. CLIP estimates how well the image matches the text.")
    clip_prompt = st.text_input("Image prompt", key="clip_prompt")
    clip_file = st.file_uploader("Upload generated image", type=["png", "jpg", "jpeg"], key="clip_image")
    if st.button("Calculate CLIP alignment", key="clip_btn"):
        if not clip_prompt.strip() or clip_file is None:
            st.warning("Provide both an image prompt and an image.")
        else:
            try:
                image = Image.open(clip_file).convert("RGB")
                with st.spinner("Loading CLIP model and calculating similarity..."):
                    score = score_image_prompt(clip_prompt, image)
                st.image(image, caption="Uploaded image", use_container_width=True)
                st.metric("CLIP similarity", f"{score:.3f}")
                st.caption("Similarity is a model score, not a guaranteed measure of image quality.")
            except Exception as e:
                st.error(f"CLIP scoring failed: {e}")

with tabs[3]:
    st.header("U4 — Run original and rewritten prompts")
    st.write("Run both prompts as text-generation requests and compare the responses.")
    original = st.text_area("Original prompt", key="u4_original",
                            value=st.session_state.get("evaluated_prompt", ""))
    rewritten = st.text_area("Rewritten prompt", key="u4_rewritten",
                             value=st.session_state.get("rewritten_prompt", ""))
    if st.button("Run both text prompts", key="run_text"):
        if not original.strip() or not rewritten.strip():
            st.warning("Provide both original and rewritten prompts.")
        else:
            try:
                with st.spinner("Running original prompt..."):
                    original_output = run_text_prompt(original)
                with st.spinner("Running rewritten prompt..."):
                    rewritten_output = run_text_prompt(rewritten)
                left, right = st.columns(2)
                left.subheader("Original output")
                left.write(original_output)
                right.subheader("Rewritten output")
                right.write(rewritten_output)
            except Exception as e:
                st.error(f"Text execution failed: {e}")
    st.divider()
    st.subheader("Image prompt execution")
    st.write("This uses Gemini image generation if the selected model is enabled for your API key.")
    image_prompt = st.text_area("Describe the image to generate", key="u4_image_prompt")
    if st.button("Generate image", key="generate_image_btn"):
        if not image_prompt.strip():
            st.warning("Enter an image prompt.")
        else:
            try:
                with st.spinner("Requesting image generation..."):
                    image_bytes = generate_image(image_prompt)
                st.image(image_bytes, caption="Generated image", use_container_width=True)
                st.download_button("Download generated image", image_bytes, "generated_image.png", "image/png")
            except Exception as e:
                st.error(f"Image generation unavailable or failed: {e}")

with tabs[4]:
    st.header("U5 — Responsible AI checks")
    st.write("Checks prompts for potentially biased wording and personal or sensitive information.")
    safety_prompt = st.text_area("Prompt to check", key="u5_prompt", height=130)
    if st.button("Run safety checks", key="safety_btn"):
        if not safety_prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            report = run_safety_checks(safety_prompt)
            st.subheader("Personal-data check")
            if report["personal_data"]["flagged"]:
                st.warning("Potential personal or sensitive data detected. Review before sharing.")
                for item in report["personal_data"]["matches"]:
                    st.write(f"- {item}")
            else:
                st.success("No patterns from the built-in personal-data checks were detected.")
            st.subheader("Bias-language check")
            if report["bias"]["flagged"]:
                st.warning("Potentially biased or exclusionary wording detected.")
                for item in report["bias"]["matches"]:
                    st.write(f"- {item}")
            else:
                st.success("No terms from the built-in bias-word list were detected.")
            st.caption("These are rule-based screening checks. They can miss sensitive data or flag harmless wording; human review is still required.")

st.divider()
st.caption("Generative AI Mini-Project P8 · Prompt Grader & Rewriter · U1–U5 starter implementation")
