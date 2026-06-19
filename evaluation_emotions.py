from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="astrosbd/french_emotion_camembert",
    top_k=None
)

import math

def weighted_cosine_similarity(res1, res2, weights=None):
    """
    res1 et res2 :
    [
        [
            {'label': 'joy', 'score': 0.8},
            ...
        ]
    ]
    """

    vec1 = {item['label']: item['score'] for item in res1[0]}
    vec2 = {item['label']: item['score'] for item in res2[0]}

    # Toutes les émotions présentes
    labels = set(vec1.keys()) | set(vec2.keys())

    # Poids par défaut = 1
    if weights is None:
        weights = {label: 1.0 for label in labels}

    numerator = 0
    norm1 = 0
    norm2 = 0

    for label in labels:
        x = vec1.get(label, 0)
        y = vec2.get(label, 0)
        w = weights.get(label, 1.0)

        numerator += w * x * y
        norm1 += w * x * x
        norm2 += w * y * y

    if norm1 == 0 or norm2 == 0:
        return 0

    return numerator / (math.sqrt(norm1) * math.sqrt(norm2))


import math

def euclidean_distance(res1, res2):
    vec1 = {item['label']: item['score'] for item in res1[0]}
    vec2 = {item['label']: item['score'] for item in res2[0]}

    # union de toutes les émotions
    labels = set(vec1.keys()) | set(vec2.keys())

    sum_squared = 0

    for label in labels:
        x = vec1.get(label, 0)
        y = vec2.get(label, 0)

        sum_squared += (x - y) ** 2

    return math.sqrt(sum_squared)

# def compute_distance(res1, res2):
#     distance = {}
    
#     vec1 = {item['label']: item['score'] for item in res1[0]}
#     res2_map = {item['label']: item['score'] for item in res2[0]}
    
#     for r in res1[0]:
#         label = r['label']
#         if label in res2_map:
#             distance[label] = r['score'] - res2_map[label]
            
#     return distance

def compute_distance(res1, res2):
    distance = {}

    res2_map = {item['label']: item['score'] for item in res2[0]}
    
    for r in res1[0]:
        label = r['label']
        if label in res2_map:
            distance[label] = math.sqrt((r['score'] - res2_map[label]) ** 2)
            
    return distance

text_ref = "\" Allez, à vélo tout le monde ! En route pour une belle balade ! \" Jules et Bou sont heureux. Jules a un nouvel ami. Et Bou a trouvé une famille. Papa et maman ont le temps de jouer avec Jules et Bou."
text_pred = "Jules est très heureux aujourd’hui. Sa famille a organisé une belle journée spéciale.Le père de Jules a préparé des vélos et la mère a préparé un pique-nique. Tous sont partis ensemble.Bou est dans le panier à vélo de maman. Il semble joyeux, avec ses babines au vent. C’est très amusant !C’est leur meilleure promenade jusqu’à présent. Ils ont passé une journée très agréable en famille : papa, maman,"
text_ori = "C'est un des plus beaux jours de sa vie ! D'autant plus que maintenant, le père de Jules a préparé les vélos et sa mère le pique-nique. Ils sont tous partis. Et devinez où est Bou… dans le panier à vélo de maman ! Il est amusant avec ses babines au vent. C'est la meilleure promenade que Jules a faite. Ils ont passé une très bonne journée tous ensemble, en famille, papa, maman, Bou et Jules !"

results_pred = classifier(text_pred)
results_ori = classifier(text_ori)
results_ref = classifier(text_ref)

print("Texte original :")
for r in results_pred[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nTexte FALC :")
for r in results_ori[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nTexte ref :")
for r in results_ref[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nDistance euclidienne entre text ori et falc predit")
dist = compute_distance(results_ori, results_pred)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(euclidean_distance(results_ori, results_pred))

print("\nDistance euclidienne entre text ori et falc ref")
dist = compute_distance(results_ori, results_ref)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(euclidean_distance(results_ori, results_ref))

print("\nDistance euclidienne entre falc pred et falc ref")
dist = compute_distance(results_pred, results_ref)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(euclidean_distance(results_pred, results_ref))