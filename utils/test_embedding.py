from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Define the example table with extended problem types
examples = [
    {"ground_truth": "Resection", "prediction": "Resectie", "problem": "Multilingual (Dutch)"},
    {"ground_truth": "Moderate", "prediction": "Mírný", "problem": "Multilingual (Czech)"},
    {"ground_truth": "Bladder", "prediction": "Head-and-neck", "problem": "Medical, semantically different"},
    {"ground_truth": "Suspicion of tumor", "prediction": "Metastasis", "problem": "Medical, semantically different"},
    {"ground_truth": "Lipoma", "prediction": "Atypical lipomatous tumour", "problem": "Medical, semantically close"},
    {"ground_truth": "Arm", "prediction": "Upper extremity", "problem": "Medical, semantically close"},
    {"ground_truth": "GIST", "prediction": "Gastro-intestinal stromal tumour", "problem": "Medical abbreviation"},
    {"ground_truth": "Renal cell carcinoma", "prediction": "RCC", "problem": "Medical abbreviation"},
    {"ground_truth": "Heart attack", "prediction": "Myocardial infarction", "problem": "Synonyms (same language)"},
    {"ground_truth": "No evidence of tumor", "prediction": "Evidence of tumor", "problem": "Negation / modifiers"},
    {"ground_truth": "CHF", "prediction": "Congestive heart failure", "problem": "Abbreviation / expansion"},
    {"ground_truth": "Liver metastases", "prediction": "Metastatic lesion in liver", "problem": "Multi-word vs single-word"},
    {"ground_truth": "Leukaemia", "prediction": "Leukemia", "problem": "Spelling variation / typo"},
    {"ground_truth": "Tumor", "prediction": "Tumeur", "problem": "Cross-lingual synonym (French)"},
    {"ground_truth": "Cold", "prediction": "Common cold", "problem": "Ambiguous term / polysemy"},
    {"ground_truth": "Stage II cancer", "prediction": "Stage 2 cancer", "problem": "Numerical entities"},
    {"ground_truth": "History of diabetes", "prediction": "No history of diabetes", "problem": "Temporal relationship / negation"},
]

# List of models to evaluate
model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/embeddinggemma-300m-medical"
]

# Initialize the results DataFrame with ground truth, prediction, problem
df = pd.DataFrame(examples)

# Loop over models
for model_name in model_names:
    print(f"Processing model: {model_name}")
    model = SentenceTransformer(model_name)
    
    scores = []
    for ex in examples:
        emb1 = model.encode(ex["ground_truth"], convert_to_tensor=True)
        emb2 = model.encode(ex["prediction"], convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2).item()
        scores.append(round(sim, 4))
    
    df[model_name.split("/")[-1]] = scores  # add column with model short name

    # Free memory
    del model
    torch.cuda.empty_cache()

# Display results
print(df)
