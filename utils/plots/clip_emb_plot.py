import datasets
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import io
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, MinMaxScaler


DATASET='NAPS'

def process_data_for_plotting(csv_path, filename_col, emotion_col, hf_df, model, processor, device):
    """
    Loads a CSV, merges with Hugging Face data, computes embeddings, and performs t-SNE.
    Returns the 2D embeddings and the corresponding labels.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return None, None
    
    df[filename_col] = df[filename_col].astype(str)
    merged_df = pd.merge(hf_df, df, left_on='filename', right_on=filename_col)

    if merged_df.empty:
        print("No matching images found between the dataset and this CSV file.")
        return None, None
    print(f"Found {len(merged_df)} matching entries.")

    def get_image_embeddings(image_list):
        embeddings = []
        with torch.no_grad():
            for item in image_list:
                try:
                    img_to_process = item
                    if isinstance(item, dict) and 'bytes' in item and item['bytes']:
                        img_to_process = Image.open(io.BytesIO(item['bytes']))
                    
                    if img_to_process.mode != 'RGB':
                        img_to_process = img_to_process.convert('RGB')
                    
                    inputs = processor(images=img_to_process, return_tensors="pt").to(device)
                    image_features = model.get_image_features(**inputs)
                    embeddings.append(image_features.cpu().numpy().flatten())
                except Exception as e:
                    embeddings.append(np.zeros(model.config.projection_dim))
        return np.array(embeddings)

    image_embeddings = get_image_embeddings(merged_df['image_x'].tolist())
    
    valid_indices = [i for i, emb in enumerate(image_embeddings) if emb.any()]
    image_embeddings = image_embeddings[valid_indices]
    merged_df = merged_df.iloc[valid_indices].reset_index(drop=True)
    
    image_embeddings_normalized = normalize(image_embeddings, norm='l2')

    perplexity_value = min(30, len(image_embeddings_normalized) - 1)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, init='pca', learning_rate='auto')

    embedding_2d = tsne.fit_transform(image_embeddings_normalized)

    tx = MinMaxScaler().fit_transform(embedding_2d[:, 0].reshape(-1, 1)).flatten()
    ty = MinMaxScaler().fit_transform(embedding_2d[:, 1].reshape(-1, 1)).flatten()
    
    return np.vstack((tx, ty)).T, merged_df[emotion_col]

try:
    dataset = datasets.load_dataset(f"filnow/{DATASET}-Emotion-Prediction", split="train")
    hf_df = dataset.to_pandas()
    hf_df['filename'] = hf_df['filename'].astype(str)
except Exception as e:
    print(f"Fatal error loading Hugging Face dataset: {e}")
    exit()

try:
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
except Exception as e:
    print(f"Fatal error loading CLIP model: {e}")
    exit()


csv_path_1 = f'../data/human_cls/{DATASET}-human-cls.csv'
filename_col_1 = 'image'
emotion_col_1 = 'true_emotion'
title_1 = 'Human Annotations (LAIGAI)'

csv_path_2 = f'../results/{DATASET}/cls/{DATASET}-cls-qwen-vl-2.5-72b.csv'
filename_col_2 = 'image'   
emotion_col_2 = 'predicted_emotion_y' 
title_2 = 'Model Predictions (Qwen-VL)'

embedding_2d_1, labels_1 = process_data_for_plotting(
    csv_path_1, filename_col_1, emotion_col_1, hf_df, model, processor, device
)
embedding_2d_2, labels_2 = process_data_for_plotting(
    csv_path_2, filename_col_2, emotion_col_2, hf_df, model, processor, device
)

if embedding_2d_1 is not None and embedding_2d_2 is not None:
    all_labels = pd.concat([labels_1, labels_2]).unique()
    le = LabelEncoder().fit(all_labels)
    
    custom_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    if len(all_labels) > len(custom_colors):
        print("Warning: More classes than custom colors. Repeating colors.")
        custom_colors = custom_colors * (len(all_labels) // len(custom_colors) + 1)
    
    color_map = {label: custom_colors[i] for i, label in enumerate(le.classes_)}

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    for i, emotion in enumerate(le.classes_):
        idx = (labels_1 == emotion)
        if np.any(idx):
            axes[0].scatter(
                embedding_2d_1[idx, 0], embedding_2d_1[idx, 1],
                color=color_map[emotion],
                label=emotion,
                alpha=0.8, s=75
            )
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].tick_params(labelbottom=False, labelleft=False)

    for i, emotion in enumerate(le.classes_):
        idx = (labels_2 == emotion)
        if np.any(idx):
            axes[1].scatter(
                embedding_2d_2[idx, 0], embedding_2d_2[idx, 1],
                color=color_map[emotion],
                label=emotion,
                alpha=0.8, s=75
            )
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].tick_params(labelbottom=False, labelleft=False)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=emotion,
                          markerfacecolor=color, markersize=12) for emotion, color in color_map.items()]
    
    num_labels = len(handles)
    max_labels_in_one_row = 6  

    if num_labels > max_labels_in_one_row:
        ncol = (num_labels + 1) // 2
    else:
        ncol = num_labels

    fig.legend(handles=handles,
               loc='lower center',             
               bbox_to_anchor=(0.5, 0.08),     
               ncol=ncol,                      
               fontsize=18,
               frameon=False)

    # Adjust titles and labels for specific datasets
    if DATASET == 'LAIGAI':
        fig.text(0.25, 0.01, '(a) LAI-GAI Human', ha='center', fontsize=30)
        fig.text(0.75, 0.01, '(b) LAI-GAI Qwen', ha='center', fontsize=30)
    elif DATASET == 'NAPS':
        fig.text(0.25, 0.01, '(c) NAPS Human', ha='center', fontsize=30)
        fig.text(0.75, 0.01, '(d) NAPS Qwen', ha='center', fontsize=30)
    elif DATASET == 'IAPS':
        fig.text(0.25, 0.01, '(e) IAPS Human', ha='center', fontsize=30)
        fig.text(0.75, 0.01, '(f) IAPS Qwen', ha='center', fontsize=30)

    fig.subplots_adjust(bottom=0.20)

    output_filename = 'embedding_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSide-by-side visualization saved as '{output_filename}'")
    plt.show()

else:
    print("\nCould not proceed with visualization due to errors in data processing.")