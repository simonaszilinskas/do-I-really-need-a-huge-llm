# AI Impact Optimizer

This small language model proposes other language models based on input complexity to lower the computational costs and environmental impact of everyday AI usage.


## Utilisation

1. Head to the following website: insert-url-here
2. Write your prompt on the bar at the bottom right of the screen and press 'send'.
3. Our model will analyse your prompt and give an adequate language model to process it. This model will be displayed on the chat, while on the right side of the screen it shows data on the environmental impact of your prompt.


## Structure

```
duels-new/
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