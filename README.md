# Do I really need a huge LLM?

This small language model proposes other language models based on input complexity to lower the computational costs and environmental impact of everyday AI usage.

## Features

- **Smart Routing**: Analyzes prompts and selects the most appropriate model
- **Energy Tracking**: Calculates energy consumption and CO₂ emissions saved
- **Cost Optimization**: Shows monetary savings compared to using high-end models
- **Real-time Statistics**: Displays cumulative savings and model usage distribution
- **Multiple Model Support**: Routes between LLMs and SLMs.

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
├── manifest.json
├── js/
│   ├── content.js
│   └── content-data.js
├── css/
│   └── styles.css
├── icons/
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
├── UI/
│   ├──app.py
│   ├──bertmodel.py
│   └──requirements.txt
└── README.md

```

## Fonctionnalités

- ✅ Activation uniquement sur l'URL spécifique
- ✅ Bouton flottant (FAB)
- ✅ Interface modale avec onglets
- ✅ Contenu pour l'impact environnemental
- ✅ Boutons interactifs (Prompt, Cartes débat, Ressources, FAQ)
- ✅ Question ultime révélable
- ✅ Architecture modulaire et extensible

## Développement

Le contenu est facilement modifiable dans le fichier `js/content.js` dans l'objet `CONTENT_DATA`.

Pour ajouter du contenu aux autres onglets, modifiez les sections correspondantes dans `CONTENT_DATA.bias` et `CONTENT_DATA.sovereignty`.

## Note

Les icônes sont actuellement des placeholders. Remplacez-les par de vraies icônes PNG aux dimensions appropriées :
- icon16.png : 16x16 pixels
- icon48.png : 48x48 pixels
- icon128.png : 128x128 pixels