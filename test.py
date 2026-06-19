import json

data = []

with open('..\data\etr-fr\sources\\test.json', 'r') as file:
    for line in file:
        if line.strip(): 
            data.append(json.loads(line))

print(f"Loaded {len(data)} items.")
# for d in data:
#     print(d['falc'])
#     print(d['original'])

# print(data[0]['falc'])
# print(data[0]['original'])

#______________________________________________________________

# import dictionnaire as dicti

# print(dicti.get_dict())

#______________________________________________________________

# import csv
# import json

# # 1. Load and compress the dictionary
# feel_dict = {}
# with open('FEEL.csv', newline='', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile, delimiter=';')
#     for row in reader:
#         word = row.pop('word')
#         # Extract emotions that are '1' (active)
#         emotions = [emo for emo, val in row.items() if val == '1']
#         if emotions:
#             feel_dict[word] = emotions

# # 2. Convert to a compact string (removes extra spaces)
# json_string = json.dumps(feel_dict, separators=(',', ':'), ensure_ascii=False)

# print(json_string)

#______________________________________________________________

# import dictionnaire

# print(dictionnaire.get_feel())

#______________________________________________________________

# from ministral_emotions import pred_falc
# import json

# data = []

# with open('..\data\etr-fr\sources\\test.json', 'r') as file:
#     for line in file:
#         if line.strip(): 
#             data.append(json.loads(line))

# src = data[0]['original']
# ref = data[0]['falc']
# pred = pred_falc(src)
# print(pred)

#______________________________________________________________

# from evaluation import evaluate_text

# metrics = evaluate_text(
#     predictions=pred,
#     references=ref,
#     sources=src,
#     lang="fr"
# )

# print(f"BLEU Score: {metrics['bleu']:.4f}")

# print(f"ROUGE-L: {metrics['rougeL']:.4f}")

# print(f"SARI Score: {metrics['sari']:.4f}")

# print("\nBERTScore:")
# print(f"Precision: {metrics['bertscore_precision']:.4f}")
# print(f"Recall:    {metrics['bertscore_recall']:.4f}")
# print(f"F1:        {metrics['bertscore_f1']:.4f}")

# print("\nReadability:")
# print(f"flesch reading ease: {metrics['fre']:.4f}")
# print(f"LIX:                 {metrics['lix']:.4f}")
# print(f"Compression Ratio:   {metrics['compression_ratio']:.4f}")

# print("\nNovelty:")
# print(f"Novelty Score: {metrics['novelty']:.4f}")

# print("\nCombined Metric:")
# print(f"SRB Score: {metrics['srb']:.4f}")

# print(f"\nNumber of Samples: {metrics['n_samples']}")

#______________________________________________________________


# from evaluation_emotions import classifier, compute_distance, euclidean_distance

# text_ref = "\" Allez, à vélo tout le monde ! En route pour une belle balade ! \" Jules et Bou sont heureux. Jules a un nouvel ami. Et Bou a trouvé une famille. Papa et maman ont le temps de jouer avec Jules et Bou."
# text_pred = "Jules est très heureux aujourd’hui. Sa famille a organisé une belle journée spéciale.Le père de Jules a préparé des vélos et la mère a préparé un pique-nique. Tous sont partis ensemble.Bou est dans le panier à vélo de maman. Il semble joyeux, avec ses babines au vent. C’est très amusant !C’est leur meilleure promenade jusqu’à présent. Ils ont passé une journée très agréable en famille : papa, maman,"
# text_ori = "C'est un des plus beaux jours de sa vie ! D'autant plus que maintenant, le père de Jules a préparé les vélos et sa mère le pique-nique. Ils sont tous partis. Et devinez où est Bou… dans le panier à vélo de maman ! Il est amusant avec ses babines au vent. C'est la meilleure promenade que Jules a faite. Ils ont passé une très bonne journée tous ensemble, en famille, papa, maman, Bou et Jules !"

# results_pred = classifier(text_pred)
# results_ori = classifier(text_ori)
# results_ref = classifier(text_ref)

# print("Texte original :")
# for r in results_pred[0]:
#     print(f"{r['label']}: {r['score']:.4f}")

# print("\nTexte FALC :")
# for r in results_ori[0]:
#     print(f"{r['label']}: {r['score']:.4f}")

# print("\nTexte ref :")
# for r in results_ref[0]:
#     print(f"{r['label']}: {r['score']:.4f}")

# print("\nDistance euclidienne entre text ori et falc predit")
# dist = compute_distance(results_ori, results_pred)
# for r in dist:
#     print(f"{r}: {dist[r]:.4f}")
# print(euclidean_distance(results_ori, results_pred))

# print("\nDistance euclidienne entre text ori et falc ref")
# dist = compute_distance(results_ori, results_ref)
# for r in dist:
#     print(f"{r}: {dist[r]:.4f}")
# print(euclidean_distance(results_ori, results_ref))

# print("\nDistance euclidienne entre falc pred et falc ref")
# dist = compute_distance(results_pred, results_ref)
# for r in dist:
#     print(f"{r}: {dist[r]:.4f}")
# print(euclidean_distance(results_pred, results_ref))





#____________________________________________________





# n = len(data)
# n = 5
# for i in range(n):
#     print(i)
#     print(data[i]['falc'])







#______________________________________________





# import numpy as np
# from transformers import pipeline
# from tqdm import tqdm

# # Load emotion classifier
# classifier = pipeline(
#     "text-classification",
#     model="astrosbd/french_emotion_camembert",
#     top_k=None
# )

# def get_emotion_vector(text):
#     outputs = classifier(text)[0]
#     outputs = sorted(outputs, key=lambda x: x['label']) # Ensure matching indices
#     return np.array([em['score'] for em in outputs])

# o = classifier(data[0]['falc'])

# print(o)

# ret = get_emotion_vector(data[0]['falc'])
# print(ret)