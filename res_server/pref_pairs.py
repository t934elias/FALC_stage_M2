# import pandas as pd
# import numpy as np

# def get_median_idx(group):
#     median_val = group['score_dpo'].median()
#     return (group['score_dpo'] - median_val).abs().idxmin()

# df_falc = pd.read_excel('.\\resultat\\summaries_for_generation.xlsx')
# df_mesure = pd.read_excel('.\\resultat\\summaries_for_evaluation.xlsx')

# meilleur = df_mesure.groupby('sentence_id')['score_dpo'].idxmax()
# pire = df_mesure.groupby('sentence_id')['score_dpo'].idxmin()
# moyen = df_mesure.groupby('sentence_id').apply(get_median_idx)

# df_meilleur = df_falc.iloc[meilleur].copy()
# df_meilleur['score_dpo'] = df_mesure.loc[meilleur, 'score_dpo']
# df_pire = df_falc.iloc[pire].copy()
# df_pire['score_dpo'] = df_mesure.loc[pire, 'score_dpo']
# df_moyen = df_falc.iloc[moyen].copy()
# df_moyen['score_dpo'] = df_mesure.loc[moyen, 'score_dpo'].values

# # print(df_meilleur)
# # print(df_pire)

# df_meilleur = df_meilleur.rename(columns={
#     'original_text': 'prompt', 
#     'falc': 'chosen'
# })

# df_pire = df_pire.rename(columns={
#     'falc': 'rejected'
# })

# df_meilleur = df_meilleur[['sentence_id', 'prompt', 'chosen', 'score_dpo_chosen']]
# df_pire = df_pire[['sentence_id', 'rejected', 'score_dpo_rejected']]

# df_dpo = pd.merge(df_meilleur, df_pire, on='sentence_id')

# final_dpo_columns = ['prompt', 'chosen', 'rejected', 'score_dpo_chosen', 'score_dpo_rejected']
# df_dpo = df_dpo[final_dpo_columns]

# # print(df_dpo.head())
# print(df_dpo)

# df_dpo.to_json('.\\resultat\\dpo_dataset.jsonl', orient='records', lines=True, force_ascii=False)



# # import pandas as pd

# # df_dpo_loaded = pd.read_json('dpo_dataset.jsonl', orient='records', lines=True)

# # print(df_dpo_loaded.head())





import pandas as pd
import numpy as np

df_falc = pd.read_excel('results/summaries_for_generation.xlsx')
df_mesure = pd.read_excel('results/summaries_for_evaluation.xlsx')

meilleur = df_mesure.groupby('sentence_id')['score_dpo'].idxmax()
pire = df_mesure.groupby('sentence_id')['score_dpo'].idxmin()

def get_median_idx(group):
    median_val = group['score_dpo'].median()
    return (group['score_dpo'] - median_val).abs().idxmin()

moyen = df_mesure.groupby('sentence_id').apply(get_median_idx)

df_meilleur = df_falc.iloc[meilleur].copy()
df_meilleur['score_dpo'] = df_mesure.loc[meilleur, 'score_dpo'].values

df_moyen = df_falc.iloc[moyen].copy()
df_moyen['score_dpo'] = df_mesure.loc[moyen, 'score_dpo'].values

df_pire = df_falc.iloc[pire].copy()
df_pire['score_dpo'] = df_mesure.loc[pire, 'score_dpo'].values

df_p1_chosen = df_meilleur.rename(columns={'original_text': 'prompt', 'falc': 'chosen', 'score_dpo': 'score_dpo_chosen'})[['sentence_id', 'prompt', 'chosen', 'score_dpo_chosen']]
df_p1_rejected = df_moyen.rename(columns={'falc': 'rejected', 'score_dpo': 'score_dpo_rejected'})[['sentence_id', 'rejected', 'score_dpo_rejected']]

df_pair_best_avg = pd.merge(df_p1_chosen, df_p1_rejected, on='sentence_id')

df_p2_chosen = df_moyen.rename(columns={'original_text': 'prompt', 'falc': 'chosen', 'score_dpo': 'score_dpo_chosen'})[['sentence_id', 'prompt', 'chosen', 'score_dpo_chosen']]
df_p2_rejected = df_pire.rename(columns={'falc': 'rejected', 'score_dpo': 'score_dpo_rejected'})[['sentence_id', 'rejected', 'score_dpo_rejected']]

df_pair_avg_worst = pd.merge(df_p2_chosen, df_p2_rejected, on='sentence_id')

df_meilleur = df_meilleur.rename(columns={'original_text': 'prompt', 'falc': 'chosen', 'score_dpo': 'score_dpo_chosen'})[['sentence_id', 'prompt', 'chosen', 'score_dpo_chosen']]
df_pire = df_pire.rename(columns={'falc': 'rejected', 'score_dpo': 'score_dpo_rejected'})[['sentence_id', 'rejected', 'score_dpo_rejected']]
df_pair_best_worst = pd.merge(df_meilleur, df_pire, on='sentence_id')

df_dpo = pd.concat([df_pair_best_worst, df_pair_best_avg, df_pair_avg_worst], ignore_index=True)

df_dpo = df_dpo[df_dpo['chosen'] != df_dpo['rejected']]

final_dpo_columns = ['prompt', 'chosen', 'rejected', 'score_dpo_chosen', 'score_dpo_rejected']
df_dpo = df_dpo[final_dpo_columns]

print(f"Total preference pairs generated: {len(df_dpo)}")
# print(df_dpo.head())

df_dpo.to_json('results/dpo_dataset.jsonl', orient='records', lines=True, force_ascii=False)