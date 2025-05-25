# Do I really need a huge LLM?

This small language model proposes other language models based on input complexity to lower the computational costs and environmental impact of everyday AI usage. This was done by training a BERT-type model into recognizing prompts as adequate for SLMs or LLM. Details on the this process are available on this repo.

This model is also available at [HuggingFace.co](https://huggingface.co/monsimas/ModernBERT-ecoRouter).
This repo was created at the {Tech: Paris} AI Hackathon on May 25th/26th 2025.

## Features

- **Smart Routing**: Analyzes prompts and selects the most appropriate model
- **Energy Tracking**: Calculates energy consumption and CO₂ emissions saved
- **Cost Optimization**: Shows monetary savings compared to using high-end models
- **Real-time Statistics**: Displays cumulative savings and model usage distribution
- **Multiple Model Support**: Routes between LLMs and SLMs.

## Data sources used

- [WildChat dataset](https://huggingface.co/datasets/allenai/WildChat)
- [ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K/tree/main/old)

## Models used

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large)
- [Mistral AI v1](https://docs.mistral.ai/api/)

## Installation

```bash
pip install -r UI/requirements.txt
```

## Usage

1. Launch the following command:

```bash
python UI/app.py
```

The interface will launch in your browser.
(add screenshot)
2. Write your prompt on the bar at the bottom right of the screen and press 'send'.

3. Our model will analyse your prompt and provide an adequate language model to process it. This model will be displayed on the chat, while data on the environmental impact of your prompt will be shown on the right side of the screen.

4. This repo also contains the methodology we used to train our model. Feel free to adapt it and retrain it to your specific needs!

## How It Works

1. **Prompt Classification**: Analyzes query complexity and type
2. **Model Selection**: Chooses the most efficient model for the task
3. **Savings Calculation**: Compares against GPT-4 baseline usage
4. **Response Generation**: Processes query with selected model

## Structure

```
do-I-really-need-a-huge-llm/
├── training/
│   ├──archive/
│   ├── classified_outputs/
│   ├── merged_prompts_input.jsonl
│   ├── mistral_predictions_final.csv
│   ├── mistral_predictions_final.jsonl
│   └── product_classification.ipynb
├── UI/
│   ├──app.py
│   ├──bertmodel.py
│   └──requirements.txt
└── README.md

```

## Who are we?

- [Simonas Zilinkas](https://github.com/simonaszilinskas)
- [Mario Rocha](https://github.com/marioluisrocha)
- [Jonas Fischer](https://github.com/JonasFischer1)
- [Charlotte Cullip](https://github.com/ccullip)
- [Amin Seffo](https://github.com/AminSeffo)

