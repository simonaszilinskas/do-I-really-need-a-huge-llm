# Do I really need a huge LLM?

This small language model proposes other language models based on input complexity to lower the computational costs and environmental impact of everyday AI usage. We accomplished this using a BERT-type model to recognize prompts as adequate for SLMs or LLMs. Details on this process are available in this repo.

Model page : https://huggingface.co/monsimas/ModernBERT-ecoRouter

Hugging Face space : https://huggingface.co/spaces/monsimas/do-i-really-need-a-huge-llm

We built this project at the {Tech: Paris} AI Hackathon on May 24th/25th, 2025.

https://github.com/user-attachments/assets/3b412e9b-af6b-4b25-855d-8c2fc00c6903



## Features

- **Smart routing**: Prompt analysis by the model router for the most appropriate model and sends the prompt to it
- **Energy tracking**: Calculation of energy consumption and costs
- **Cost optimization**: Displays monetary savings or expenditures
- **Real-time statistics**: Displays cumulative savings

## Data sources used

- We trained our data using the [WildChat dataset](https://huggingface.co/datasets/allenai/WildChat) and [ShareGPT52K dataset](https://huggingface.co/datasets/RyokoAI/ShareGPT52K/tree/main/old). Labels were assigned using [mistral-large-latest](https://mistral.ai/news/mistral-large).
- Environmental impact from the models is estimated using [EcoLogits](https://huggingface.co/spaces/genai-impact/ecologits-calculator) and the monetary costs are calculated based on data from [OpenRouter.ai](https://openrouter.ai/).

## Installation

```bash
pip install -r UI/requirements.txt
```

## Usage

1. Set .env variable for Open Router
2. Launch the following command:

```bash
python UI/app.py
```

2. Write your prompt in the search bar at the bottom left of the screen and press 'send'.

3. Our router model will analyze your prompt and route towards an adequate language model to process it. This model will be displayed in the chat, while the interface will show data on your prompt's environmental impact and computational costs on the screen's right side.

4. This repo also contains the methodology we used to train our model. Feel free to adapt it and retrain it to your specific needs!

## Structure

```
do-I-really-need-a-huge-llm/
├── images/
│   └── screenshot.jpg
├── preprocessing/
│   ├── data/
│   │   ├── separated_test_data.json
│   │   ├── separated_train_data.json
│   │   └── separated_validation_data.json
│   ├── classified_outputs/
│   ├── merged_prompts_input.jsonl
│   ├── mistral_predictions_final.csv
│   ├── mistral_predictions_final.jsonl
│   └── product_classification.ipynb
├── UI/
│   ├── app.py
│   ├── bertmodel.py
│   └── requirements.txt
└── README.md

```

## Who are we?

- [Simonas Zilinkas](https://github.com/simonaszilinskas)
- [Mario Rocha](https://github.com/marioluisrocha)
- [Jonas Fischer](https://github.com/JonasFischer1)
- [Charlotte Cullip](https://github.com/ccullip)
- [Amin Seffo](https://github.com/AminSeffo)

