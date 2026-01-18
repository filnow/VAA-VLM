import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE_PATH = "utils/LAIGAI-context-vs-reg-cohens-d.csv" 

try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file was not found at '{CSV_FILE_PATH}'")
    exit()

name_mapping = {
    "Amusement": "Amusement",
    "Anger": "Anger",
    "Attachment love": "Attachment Love",
    "Awe": "Awe",
    "Craving": "Craving",
    "Disgust": "Disgust",
    "Excitement": "Excitement",
    "Fear": "Fear",
    "Joy": "Joy",
    "Neutral": "Neutral",
    "Nurturant love": "Nurturant Love",
    "Sadness": "Sadness",
    "Positive": "Positive",
    "Negative": "Negative",
    "Calm": "Calm",
    "Aroused": "Aroused",
    "Approach": "Motivated to Approach",
    "Avoid": "Motivated to Avoid"
}

custom_order = [
    "Amusement",  
    "Anger", 
    "Attachment Love",  
    "Awe",
    "Craving", 
    "Disgust",
    "Excitement",
    "Fear",
    "Joy",
    "Neutral",
    "Nurturant Love",
    "Sadness",
    "Positive",
    "Negative",
    "Calm",
    "Aroused",
    "Motivated to Approach",
    "Motivated to Avoid"
]

df['emotion'] = df['emotion'].map(name_mapping)
df['emotion'] = pd.Categorical(df['emotion'], categories=custom_order, ordered=True)

df = df.sort_values('emotion', ascending=False)

error = [df['cohens_d'] - df['ci_95_lower'], 
         df['ci_95_upper'] - df['cohens_d']]

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))
df['significant'] = ~((df['ci_95_lower'] < 0) & (df['ci_95_upper'] > 0))

def get_color(row):
    if not row['significant']:
        return 'lightgray'
    return '#007ACC' if row['cohens_d'] > 0 else '#D62728'

colors = df.apply(get_color, axis=1)

ax.errorbar(
    x=df['cohens_d'], 
    y=df['emotion'], 
    xerr=error,
    fmt='o',
    ecolor='gray',
    elinewidth=1.5,
    capsize=4,
    markersize=0 
)

ax.scatter(df['cohens_d'], df['emotion'], c=colors, s=80, zorder=10)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle=':', linewidth=0.5)
ax.grid(axis='y', which='major', linestyle='-', linewidth=0.5, color='#EEEEEE')
ax.text(0.98, 0.99, 'Context was beneficial →', transform=ax.transAxes, fontsize=11, ha='right', va='top', color='#007ACC', weight='bold')
ax.text(0.02, 0.99, '← Context was detrimental', transform=ax.transAxes, fontsize=11, ha='left', va='top', color='#D62728', weight='bold')

plt.tight_layout()
plt.savefig('cohens_d_accuracy_effect_final.png', dpi=300, bbox_inches='tight')