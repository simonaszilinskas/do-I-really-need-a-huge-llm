# Eco-Efficient AI Router

An intelligent chatbot interface that routes queries to the most efficient AI model, saving energy and costs while maintaining quality responses.

## Features

- **Smart Routing**: Analyzes prompts and selects the most appropriate model
- **Energy Tracking**: Calculates energy consumption and CO₂ emissions saved
- **Cost Optimization**: Shows monetary savings compared to using high-end models
- **Real-time Statistics**: Displays cumulative savings and model usage distribution
- **Multiple Model Support**: Routes between GPT-4, GPT-3.5, Claude Instant, Llama 7B, and search engines

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

The interface will launch in your browser. Start chatting and watch as the system intelligently routes your queries while tracking environmental and cost savings.

## How It Works

1. **Prompt Classification**: Analyzes query complexity and type
2. **Model Selection**: Chooses the most efficient model for the task
3. **Savings Calculation**: Compares against GPT-4 baseline usage
4. **Response Generation**: Processes query with selected model

## Model Selection Logic

- **Search Engine**: Current events, factual queries, real-time data
- **Llama 7B**: Basic Q&A, simple tasks
- **Claude Instant**: Quick responses, basic analysis
- **GPT-3.5**: General Q&A, simple coding, summarization
- **GPT-4**: Complex reasoning, advanced coding, creative writing