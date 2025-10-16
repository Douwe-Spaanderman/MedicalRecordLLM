from matplotlib.colors import LinearSegmentedColormap

linewidth = 1
fontsize = 18
subfontsize = 18
tickfontsize = 18
edgecolor = "black"
errorbar_color = "black"
style = "ticks"
barwidth = 0.8

custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": edgecolor,
    "patch.linewidth": linewidth,
    "patch.edgecolor": edgecolor,
}

# --- Custom Style Parameters ---
palette = ['#66c2a5','#fc8d62','#8da0cb', '#a6d854', '#e78ac3']
white_to_red = LinearSegmentedColormap.from_list("white_red", ["#F2F0EF", "#dd4040"])
red_to_white = LinearSegmentedColormap.from_list("white_red", ["#dd4040", "#F2F0EF"])

category_colors = {
    'large': palette[0],
    'medium': palette[1],
    'small': palette[2],
    'tiny': palette[3],
    'specialized': palette[4]
}

prompting_strategy_markers = {
    'ZeroShot': {"marker": ' ', 'name': 'Zero Shot'},
    'OneShot': {"marker": 'v', 'name': 'One Shot', 'symbol': '▼'},
    'FewShot': {"marker": 's', 'name': 'Few Shot', 'symbol': '■'},
    'CoT': {"marker": '^', 'name': 'Chain-of-Thought', 'symbol': '▲'},
    'SelfConsistency': {"marker": 'D', 'name': 'Self-Consistency', 'symbol': '◆'},
    'PromptGraph': {"marker": 'o', 'name': 'Prompt Graph', 'symbol': '●'},
}

model_sizes = {
    'DeepSeek-R1-0528': {'category': 'large', 'size': 685, 'name': 'DeepSeek R1 0528'},
    'Llama-4-Maverick-17B-128E-Instruct': {'category': 'large', 'size': 402, 'name': 'Llama-4 Maverick 17B 128E Instruct'},
    'Qwen3-235B-A22B': {'category': 'large', 'size': 235, 'name': 'Qwen3 235B A22B'},
    'Llama3-Med42-70B': {'category': 'specialized', 'size': 70, 'name': 'Llama-3 Med42 70B'}, 
    'Llama-3_3-Nemotron-Super-49B-v1': {'category': 'medium', 'size': 49, 'name': 'Llama-3.3 Nemotron Super 49B v1'},
    'Llama-4-Scout-17B-16E': {'category': 'medium', 'size': 109, 'name': 'Llama-4 Scout 17B 16E'}, 
    'Qwen2.5-72B-Instruct': {'category': 'medium', 'size': 72, 'name': 'Qwen2.5 72B Instruct'}, 
    'Llama3-OpenBioLLM-70B': {'category': 'specialized', 'size': 70, 'name': 'Llama-3 OpenBioLLM 70B'}, 
    'medgemma-27b-it': {'category': 'specialized', 'size': 27, 'name': 'MedGemma 27B IT'}, 
    'gemma-3-27b-it': {'category': 'small', 'size': 27, 'name': 'Gemma 3 27B IT'},
    'Mistral-Small-3.1-24B-Instruct-2503': {'category': 'small', 'size': 24, 'name': 'Mistral Small 3.1 24B Instruct 2503'}, 
    'DeepSeek-R1-0528-Qwen3-8B': {'category': 'small', 'size': 8, 'name': 'DeepSeek R1 0528 Qwen3 8B'},
    'gemma-3-4b-it': {'category': 'tiny', 'size': 4, 'name': 'Gemma 3 4B IT'},
    'Qwen3-1.7B': {'category': 'tiny', 'size': 1.7, 'name': 'Qwen3 1.7B'},
    'DeepSeek-R1-Distill-Qwen-1.5B': {'category': 'tiny', 'size': 1.5, 'name': 'DeepSeek R1 Distill Qwen 1.5B'},
}

use_cases = {
    "CRLM": "Colorectal Liver Metastases (Dutch)",
    "Liver": "Liver Tumours (Dutch)",
    "Dementia": "Dementia (Dutch)",
    "STT_English": "Soft Tissue Tumours (English)",
    "STT_Dutch": "Soft Tissue Tumours (Dutch)",
    "Melanoma": "Melanoma (Dutch)",
    "Sarcoma_Czech": "Sarcoma (Czech)"
}

figsizes = {
    "CRLM": (28, 40),
    "Liver": (44, 40),
    "Dementia": (54, 40),
    "STT_English": (38, 40),
    "STT_Dutch": (38, 40),
    "Melanoma": (46, 40),
    "Sarcoma_Czech": (37, 40)
}