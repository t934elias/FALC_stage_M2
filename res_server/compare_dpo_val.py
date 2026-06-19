import pandas as pd
import numpy as np

df_falc_dpo = pd.read_csv('results/summaries_for_generation_dpo.csv')
df_mesure_dpo = pd.read_csv('results/summaries_for_evaluation_dpo.csv')

df_falc_val = pd.read_csv('results/summaries_for_generation_val.csv')
df_mesure_val = pd.read_csv('results/summaries_for_evaluation_val.csv')

meilleur_val = df_mesure_val.groupby('sentence_id')['score_dpo'].idxmax()

df_meilleur_val = df_falc_val.iloc[meilleur_val].copy()
df_meilleur_val['score_dpo'] = df_mesure_val.loc[meilleur_val, 'score_dpo'].values


meilleur_dpo = df_mesure_dpo.groupby('sentence_id')['score_dpo'].idxmax()

df_meilleur_dpo = df_falc_dpo.iloc[meilleur_dpo].copy()
df_meilleur_dpo['score_dpo'] = df_mesure_dpo.loc[meilleur_dpo, 'score_dpo'].values




df_comparison = pd.merge(
    df_meilleur_val, 
    df_meilleur_dpo, 
    on='sentence_id', 
    suffixes=('_val', '_dpo')
)

df_comparison['better_model'] = np.select(
    [
        df_comparison['score_dpo_val'] > df_comparison['score_dpo_dpo'],
        df_comparison['score_dpo_val'] < df_comparison['score_dpo_dpo']
    ],
    ['val', 'dpo'],
    default='tie'
)

print("--- Win / Loss / Tie Count ---")
print(df_comparison['better_model'].value_counts())

print("\n--- Percentage Share ---")
print(df_comparison['better_model'].value_counts(normalize=True) * 100)

# 6. Save the final comparison sheet
df_comparison.to_csv('results/val_dpo_comparison.csv', index=False)