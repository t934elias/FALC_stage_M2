import textstat
from evaluate import load
from bert_score import scorer
from transformers import pipeline
import numpy as np




# this is like evaluation falc
sari_metric = load("sari")
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

def calculate_etr_metrics(original, generated, reference):
    metrics = {}
    
    sari_res = sari_metric.compute(sources=[original], predictions=[generated], references=[[reference]])
    metrics['SARI'] = sari_res['sari'] / 100.0 # Normalize to 0-1
    
    P, R, F1 = scorer.score([generated], [reference], lang="en", model_type="microsoft/deberta-xlarge-mnli")
    metrics['BERTScore'] = F1.item()
    
    orig_emotions = {res['label']: res['score'] for res in emotion_classifier(original)[0]}
    gen_emotions = {res['label']: res['score'] for res in emotion_classifier(generated)[0]}
    
    v1 = np.array([orig_emotions.get(k, 0) for k in sorted(orig_emotions.keys())])
    v2 = np.array([gen_emotions.get(k, 0) for k in sorted(gen_emotions.keys())])
    metrics['Emotional_Similarity'] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    metrics['FRE'] = max(0.0, min(100.0, textstat.flesch_reading_ease(generated))) / 100.0
    
    # Let's map it so 1 is perfectly simple (<20) and 0 is too complex (>60).
    lix_score = textstat.lix(generated)
    metrics['LIX_Simplicity'] = max(0.0, min(1.0, (60 - lix_score) / 40))

    # Target an ideal compression ratio (e.g., 0.75 of original length)
    comp_ratio = len(generated.split()) / max(1, len(original.split()))
    metrics['Compression_Fit'] = 1.0 - abs(0.75 - comp_ratio) # Penalizes if too long OR too empty
    
    # Novelty: Words in generated that weren't in original (lexical substitution)
    orig_words = set(original.lower().split())
    gen_words = set(generated.lower().split())
    added_words = gen_words - orig_words
    metrics['Novelty'] = len(added_words) / max(1, len(gen_words))
    
    return metrics

# Example Execution:
orig = "The configuration of architectural constructs necessitates an unyielding dedication to equilibrium."
gen = "Building houses requires a strong focus on balance."
ref = "When you build houses, you must focus on balance."

scores = calculate_etr_metrics(orig, gen, ref)
for k, v in scores.items():
    print(f"{k}: {v:.3f}")













# this is like compare_falc2
import textstat
from evaluate import load
from bert_score import scorer
import numpy as np

# Load the SARI metric from Hugging Face evaluate
# Note: SARI natively expects a reference, so for the baseline test 
# we use the Original Text as its own reference to measure structural change.
sari_metric = load("sari")

def evaluate_text_quality(original, target):
    """
    Evaluates a target text (either Human or Model) against the Original Text
    to establish an absolute baseline of quality and readability.
    """
    metrics = {}
    
    # --- 1. Readability Scores (Higher FRE / Lower LIX = Better) ---
    metrics['FRE'] = textstat.flesch_reading_ease(target)
    metrics['LIX'] = textstat.lix(target)
    
    # --- 2. Simplification Mechanics (SARI) ---
    # We pass the original as the source, target as prediction, and original as reference
    # to see how effectively the text was structurally altered/simplified.
    sari_res = sari_metric.compute(sources=[original], predictions=[target], references=[[original]])
    metrics['SARI'] = sari_res['sari']
    
    # --- 3. Compression Fit ---
    # Measures how much fluff was cut. ETR target is usually ~70-75% of original length.
    orig_len = len(original.split())
    target_len = len(target.split())
    metrics['Compression_Ratio'] = target_len / max(1, orig_len)
    
    return metrics

def compare_model_vs_human(original_texts, model_texts, human_texts):
    """
    Loops through your dataset, evaluates both model and human against the original,
    and calculates the average scores to find the winner.
    """
    model_totals = {'FRE': [], 'LIX': [], 'SARI': [], 'Compression_Ratio': []}
    human_totals = {'FRE': [], 'LIX': [], 'SARI': [], 'Compression_Ratio': []}
    
    for orig, model, human in zip(original_texts, model_texts, human_texts):
        # Evaluate Model
        m_stats = evaluate_text_quality(orig, model)
        for k, v in m_stats.items():
            model_totals[k].append(v)
            
        # Evaluate Human
        h_stats = evaluate_text_quality(orig, human)
        for k, v in h_stats.items():
            human_totals[k].append(v)
            
    # Compute Averages
    print(f"{'Metric':<20} | {'Human Average':<15} | {'Model Average':<15} | {'Winner':<10}")
    print("-" * 68)
    
    for metric in model_totals.keys():
        avg_human = np.mean(human_totals[metric])
        avg_model = np.mean(model_totals[metric])
        
        # Determine winner based on metric rules
        if metric == 'FRE' or metric == 'SARI':
            winner = "Model AI" if avg_model > avg_human else "Human"
        elif metric == 'LIX':
            winner = "Model AI" if avg_model < avg_human else "Human" # Lower LIX is easier
        elif metric == 'Compression_Ratio':
            # Closer to ideal 0.75 target wins
            winner = "Model AI" if abs(0.75 - avg_model) < abs(0.75 - avg_human) else "Human"
            
        print(f"{metric:<20} | {avg_human:<15.2f} | {avg_model:<15.2f} | {winner:<10}")

# --- EXAMPLE DATASET RUN ---
if __name__ == "__main__":
    # Sample parallel data
    originals = [
        "The configuration of architectural constructs necessitates an unyielding dedication to equilibrium.",
        "In accordance with recent legislative updates, citizens are mandated to report earnings punctually."
    ]
    
    models = [
        "Building houses requires a strong focus on balance.",
        "New laws mean you must report your income on time."
    ]
    
    humans = [
        "When you build houses, you must always make sure that they are balanced and stable.",
        "Because of new updates to the law, all citizens have to report their money on time."
    ]
    
    compare_model_vs_human(originals, models, humans)