import os
import ctypes
from functools import reduce
import nltk
import numpy as np
import spacy
import textacy.extract
import textacy.preprocessing
import textdescriptives as td
import evaluate
from scipy.stats import hmean

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    ctypes.CDLL(r"E:\python_envs\etr_env\Lib\site-packages\torch\lib\libiomp5md.dll")
except Exception as e:
    print(f"Manual load failed: {e}")

nltk.download("punkt")

METRIC_KEY_ROUGE = "rouge"
METRIC_KEY_ROUGEL = "rougeL"
METRIC_KEY_BLEU = "bleu"
METRIC_KEY_BERTSCORE_F1 = "bertscore_f1"
METRIC_KEY_KMRE = "fre"
METRIC_KEY_LIX = "lix"
METRIC_KEY_COMPRESSION = "compression_ratio"
METRIC_KEY_NOVELTY = "novelty"
METRIC_KEY_SRB = "srb"
METRIC_KEY_SARI = "sari"


def load_nlp(lang="fr"):
    spacy_models = {
        "en": "en_core_web_md",
        "fr": "fr_core_news_md"
    }
    nlp = spacy.load(spacy_models[lang])
    nlp.add_pipe("textdescriptives/readability")

    return nlp

def compute_readability_scores(texts, nlp):

    docs = nlp.pipe(texts)

    metrics_list = td.extract_dict(
        docs,
        include_text=False
    )

    merged_dict = reduce(
        lambda x, y: {**x, **y},
        metrics_list
    )

    return {
        k: [merged_dict[k] for merged_dict in metrics_list]
        for k in merged_dict
    }

def compute_bleu(predictions, references):

    bleu_metric = evaluate.load("sacrebleu")

    preds = [p.strip() for p in predictions]

    refs = [
        [r[0].strip()] if isinstance(r, list)
        else [r.strip()]
        for r in references
    ]

    result = bleu_metric.compute(
        predictions=preds,
        references=refs
    )

    return {
        METRIC_KEY_BLEU: result["score"]
    }

def compute_rouge(predictions, references):

    rouge_metric = evaluate.load("rouge")

    def preprocess(texts):
        return [
            "\n".join(nltk.sent_tokenize(text.strip()))
            for text in texts
        ]

    preds = preprocess(predictions)

    refs = preprocess([
        r[0] if isinstance(r, list) else r
        for r in references
    ])

    result = rouge_metric.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True
    )

    return {
        k: round(v * 100, 4)
        for k, v in result.items()
    }

def compute_sari(predictions, references, sources):

    sari_metric = evaluate.load("sari")

    preds = [p.strip() for p in predictions]

    refs = [
        [r[0].strip()] if isinstance(r, list)
        else [r.strip()]
        for r in references
    ]

    srcs = [s.strip() for s in sources]

    result = sari_metric.compute(
        predictions=preds,
        references=refs,
        sources=srcs
    )

    return {
        METRIC_KEY_SARI: result["sari"]
    }

def compute_bertscore(predictions, references, lang="fr"):

    bertscore_metric = evaluate.load("bertscore")

    preds = [p.strip() for p in predictions]

    refs = [
        r[0].strip() if isinstance(r, list)
        else r.strip()
        for r in references
    ]

    result = bertscore_metric.compute(
        predictions=preds,
        references=refs,
        lang=lang
    )

    return {
        "bertscore_precision":
            np.mean(result["precision"]) * 100,

        "bertscore_recall":
            np.mean(result["recall"]) * 100,

        METRIC_KEY_BERTSCORE_F1:
            np.mean(result["f1"]) * 100,
    }

def compute_readability(predictions, sources, nlp):

    preds = [p.strip() for p in predictions]
    srcs = [s.strip() for s in sources]

    scores_preds = compute_readability_scores(
        preds,
        nlp
    )

    scores_sources = compute_readability_scores(
        srcs,
        nlp
    )

    def compression_ratio(ntok_src, ntok_tgt):
        return 100 - (
            ntok_tgt / ntok_src * 100
        )

    # print(scores_preds)
    return {
        

        METRIC_KEY_KMRE:
            np.mean(
                scores_preds["flesch_reading_ease"]
            ),

        METRIC_KEY_LIX:
            np.mean(
                scores_preds["lix"]
            ),

        METRIC_KEY_COMPRESSION:
            np.mean(
                compression_ratio(
                    np.array(scores_sources["n_tokens"]),
                    np.array(scores_preds["n_tokens"])
                )
            ),
    }

def compute_novelty(predictions, sources, nlp):

    def unigram(text):

        text = nlp(
            textacy.preprocessing.remove.accents(
                text.lower()
            )
        )

        return set([
            span.text
            for span in textacy.extract.ngrams(
                text,
                1,
                filter_nums=True
            )
        ])

    def novelty(source, prediction):

        src_unigrams = unigram(source)
        pred_unigrams = unigram(prediction)

        try:
            return (
                len(pred_unigrams - src_unigrams)
                / len(pred_unigrams)
                * 100
            )

        except:
            return 0.0

    scores = [
        novelty(s, p)
        for s, p in zip(sources, predictions)
    ]

    return {
        METRIC_KEY_NOVELTY: np.mean(scores)
    }

def compute_srb(sari, rougeL, bertscore_f1):

    return {
        METRIC_KEY_SRB:
            hmean([
                sari,
                rougeL,
                bertscore_f1
            ])
    }

def evaluate_text(
    predictions,
    references,
    sources=None,
    lang="fr"
):

    results = {}
    nlp = load_nlp(lang)

    results.update(
        compute_bleu(
            predictions,
            references
        )
    )

    results.update(
        compute_rouge(
            predictions,
            references
        )
    )

    if sources is not None:

        results.update(
            compute_sari(
                predictions,
                references,
                sources
            )
        )

    results.update(
        compute_bertscore(
            predictions,
            references,
            lang
        )
    )

    if sources is not None:

        results.update(
            compute_readability(
                predictions,
                sources,
                nlp
            )
        )

    if sources is not None:

        results.update(
            compute_novelty(
                predictions,
                sources,
                nlp
            )
        )

    if sources is not None:

        results.update(
            compute_srb(
                results[METRIC_KEY_SARI],
                results[METRIC_KEY_ROUGEL],
                results[METRIC_KEY_BERTSCORE_F1]
            )
        )

    results["n_samples"] = len(predictions)

    final_results = {}

    for k, v in results.items():

        if isinstance(v, (float, np.floating)):
            final_results[k] = round(v, 4)
        else:
            final_results[k] = v

    return final_results


# # src = ["The cat sat on the mat which was very old and dusty."] # original
# # pred = ["The cat sat on the old mat."] # modele
# # ref = [["The cat sat on the dusty mat.", "The cat sat on the old rug."]] # what it should be

# # pred = ["C'est un très beau jour pour Jules ! Son papa a préparé les vélos. Sa maman a fait un pique-nique. Ils partent tous ensemble. Bou est dans le panier de maman. Il rit avec le vent dans la bouche. C'est la plus belle journée de Jules ! Ils jouent ensemble : papa, maman, Bou et Jules. C'est une journée en famille, joyeuse et pleine de bonheur."]
# src = ["C'est un des plus beaux jours de sa vie ! D'autant plus que maintenant, le père de Jules a préparé les vélos et sa mère le pique-nique. Ils sont tous partis. Et devinez où est Bou… dans le panier à vélo de maman ! Il est amusant avec ses babines au vent. C'est la meilleure promenade que Jules a faite. Ils ont passé une très bonne journée tous ensemble, en famille, papa, maman, Bou et Jules !"]
# ref = [["\" Allez, à vélo tout le monde ! En route pour une belle balade ! \" Jules et Bou sont heureux. Jules a un nouvel ami. Et Bou a trouvé une famille. Papa et maman ont le temps de jouer avec Jules et Bou."]]
# pred = ["Jules est très heureux aujourd’hui. Sa famille a organisé une belle journée spéciale.Le père de Jules a préparé des vélos et la mère a préparé un pique-nique. Tous sont partis ensemble.Bou est dans le panier à vélo de maman. Il semble joyeux, avec ses babines au vent. C’est très amusant !C’est leur meilleure promenade jusqu’à présent. Ils ont passé une journée très agréable en famille : papa, maman,"]


src = ["C'est un des plus beaux jours de sa vie ! D'autant plus que maintenant, le père de Jules a préparé les vélos et sa mère le pique-nique. Ils sont tous partis. Et devinez où est Bou… dans le panier à vélo de maman ! Il est amusant avec ses babines au vent. C'est la meilleure promenade que Jules a faite. Ils ont passé une très bonne journée tous ensemble, en famille, papa, maman, Bou et Jules !"]
ref = [["\" Allez, à vélo tout le monde ! En route pour une belle balade ! \" Jules et Bou sont heureux. Jules a un nouvel ami. Et Bou a trouvé une famille. Papa et maman ont le temps de jouer avec Jules et Bou."]]
pred = ["Jules est très heureux aujourd’hui. Sa famille a organisé une belle journée spéciale.Le père de Jules a préparé des vélos et la mère a préparé un pique-nique. Tous sont partis ensemble.Bou est dans le panier à vélo de maman. Il semble joyeux, avec ses babines au vent. C’est très amusant !C’est leur meilleure promenade jusqu’à présent. Ils ont passé une journée très agréable en famille : papa, maman,"]


metrics = evaluate_text(
    predictions=pred,
    references=ref,
    sources=src,
    lang="fr"
)

print(f"BLEU Score: {metrics['bleu']:.4f}")

print(f"ROUGE-L: {metrics['rougeL']:.4f}")

print(f"SARI Score: {metrics['sari']:.4f}")

print("\nBERTScore:")
print(f"Precision: {metrics['bertscore_precision']:.4f}")
print(f"Recall:    {metrics['bertscore_recall']:.4f}")
print(f"F1:        {metrics['bertscore_f1']:.4f}")

print("\nReadability:")
print(f"flesch reading ease: {metrics['fre']:.4f}")
print(f"LIX:                 {metrics['lix']:.4f}")
print(f"Compression Ratio:   {metrics['compression_ratio']:.4f}")

print("\nNovelty:")
print(f"Novelty Score: {metrics['novelty']:.4f}")

print("\nCombined Metric:")
print(f"SRB Score: {metrics['srb']:.4f}")

print(f"\nNumber of Samples: {metrics['n_samples']}")