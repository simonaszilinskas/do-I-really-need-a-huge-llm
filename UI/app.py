import gradio as gr
import json
import time
from typing import Dict, Tuple, List
from bertmodel import predict_label
import tiktoken
# from ecologits import EcoLogits  # Removed - using OpenRouter instead
# from openai import OpenAI  # Removed - using OpenRouter instead
from dotenv import load_dotenv
import os
import requests
import json

# Set environment variable to suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Model configurations with energy consumption and cost estimates
MODEL_CONFIGS = {
    "large": {
        "name": "Claude Opus 4",
        "energy_per_token": 0.0001,  # kWh per token (estimated)
        "cost_per_token": 0.00003,   # USD per token
        "icon": "🧠"
    },
    "small": {
        "name": "Mistral Small 24B",
        "energy_per_token": 0.00002,
        "cost_per_token": 0.0000005,
        "icon": "⚡"
    }
}

class ModelRouter:
    def __init__(self):
        self.routing_history = []
        print("[INIT] ModelRouter initialized")
    
    def classify_prompt(self, prompt: str) -> str:
        print(f"\n[CLASSIFY] Classifying prompt: '{prompt[:50]}...'")
        label = predict_label(prompt)
        print(f"[CLASSIFY] ModernBERT returned label: '{label}'")
        return label
    
    def select_model(self, prompt: str) -> str:
        """Select the most efficient model based on prompt classification."""
        prompt_type = self.classify_prompt(prompt)
        # Normalize
        key = prompt_type.strip().lower()
        print(f"[SELECT] Normalized label: '{key}'")

        # Map normalized labels to actual MODEL_CONFIGS keys
        if "small" in key:
            print(f"[SELECT] Selected: SMALL model (Mistral Small 24B)")
            return "small"
        else:
            print(f"[SELECT] Selected: LARGE model (Claude Opus 4)")
            return "large"
    

    def estimate_tokens(self, 
                        prompt: str, 
                        response: str | None = None,
                        max_response_tokens: int | None = None) -> int:
        """
        Estimate total token count: exact prompt tokens + 
        a target number of response tokens.
        """
        model_name = "gpt-4"
        encoding = tiktoken.encoding_for_model(model_name)
        # 1) exact prompt tokenization
        prompt_tokens = len(encoding.encode(prompt))
        print(f"[TOKENS] Prompt tokens: {prompt_tokens}")

        if response is not None:
            response_tokens = len(encoding.encode(response))
        elif max_response_tokens is not None:
            # you’re reserving this many tokens for the model’s reply
            response_tokens = max_response_tokens
        else:
            # fallback to a default budget
            response_tokens = 0

        total_tokens = prompt_tokens + response_tokens
        print(f"[TOKENS] Response tokens: {response_tokens}, Total: {total_tokens}")
        return total_tokens
    
    def estimate_large_model_energy(self, tokens: int) -> float:
        """
        Estimate large model energy consumption based on tokens.
        Using empirical estimates for energy consumption.
        """
        large_config = MODEL_CONFIGS["large"]
        return tokens * large_config["energy_per_token"]
    
    def calculate_savings(self, selected_model: str, prompt: str) -> Dict:
        """Calculate energy and cost savings compared to using the large model"""
        print(f"[SAVINGS] Calculating for model: {selected_model}")
        tokens = self.estimate_tokens(prompt)
        
        selected_config = MODEL_CONFIGS[selected_model]
        large_config = MODEL_CONFIGS["large"]
        
        # Calculate actual usage
        actual_energy = tokens * selected_config["energy_per_token"]
        actual_cost = tokens * selected_config["cost_per_token"]
        
        # Calculate large model usage
        large_energy = self.estimate_large_model_energy(tokens)
        large_cost = tokens * large_config["cost_per_token"]
        
        # Calculate savings (only positive if small model is selected)
        if selected_model == "small":
            energy_saved = large_energy - actual_energy
            cost_saved = large_cost - actual_cost
            energy_saved_percent = (energy_saved / large_energy) * 100 if large_energy > 0 else 0
            cost_saved_percent = (cost_saved / large_cost) * 100 if large_cost > 0 else 0
        else:
            # No savings if using the large model
            energy_saved = 0
            cost_saved = 0
            energy_saved_percent = 0
            cost_saved_percent = 0
        
        return {
            "selected_model": selected_config["name"],
            "tokens": tokens,
            "actual_energy": actual_energy,
            "actual_cost": actual_cost,
            "large_energy": large_energy,
            "large_cost": large_cost,
            "energy_saved": energy_saved,
            "cost_saved": cost_saved,
            "energy_saved_percent": energy_saved_percent,
            "cost_saved_percent": cost_saved_percent,
            "co2_saved_grams": energy_saved * 400  # Approximate CO2 per kWh
        }

print("[STARTUP] Initializing ModelRouter...")
router = ModelRouter()
print("[STARTUP] ModelRouter ready")
print(f"[STARTUP] Available models: {list(MODEL_CONFIGS.keys())}")
print(f"[STARTUP] OpenRouter API Key: {'SET' if OPENROUTER_API_KEY else 'NOT SET'}")

def process_message(message: str, history: List[List[str]]) -> Tuple[str, str, str]:
    """Process the user message and return response with savings info"""
    print(f"\n{'='*60}")
    print(f"[PROCESS] New message received: '{message[:100]}...'")
    
    # Route to appropriate model
    selected_model = router.select_model(message)
    model_config = MODEL_CONFIGS[selected_model]
    print(f"[PROCESS] Using model config: {model_config['name']}")
    
    # Calculate savings
    print(f"[PROCESS] Calculating savings...")
    savings = router.calculate_savings(selected_model, message)
    print(f"[PROCESS] Savings calculated: {savings['energy_saved_percent']:.1f}% energy, {savings['cost_saved_percent']:.1f}% cost")
    
    open_router_model_dict = {
        "large": "anthropic/claude-opus-4",
        "small": "mistralai/mistral-small-24b-instruct-2501"
    }
    # Check if API key is available
    if not OPENROUTER_API_KEY:
        print(f"[API] No OpenRouter API key found - running in DEMO MODE")
        answer = f"[Demo Mode] This would be a response from {model_config['name']} to: {message[:50]}..."
    else:
        print(f"[API] OpenRouter API key found: {OPENROUTER_API_KEY[:10]}...")
        try:
            model_id = open_router_model_dict[selected_model]
            print(f"[API] Calling OpenRouter with model: {model_id}")
            
            request_data = {
                "model": model_id,
                "messages": [
                {
                    "role": "user",
                    "content": message
                }
                ]
            }
            print(f"[API] Request data: {json.dumps(request_data, indent=2)[:200]}...")
            
            response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps(request_data)
            )
            
            # Debug: print response status and content
            print(f"[API] Response Status Code: {response.status_code}")
            print(f"[API] Response Headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"[API ERROR] Full response: {response.text}")
                answer = f"[API Error {response.status_code}] {response.text[:200]}..."
            else:
                data = response.json()
                print(f"[API] Response keys: {list(data.keys())}")
                
                if "choices" in data and len(data["choices"]) > 0:
                    answer = data["choices"][0]["message"]["content"]
                    print(f"[API] Successfully got response: {answer[:100]}...")
                else:
                    print(f"[API ERROR] Unexpected response format: {json.dumps(data, indent=2)}")
                    answer = f"[Error] Unexpected response format from OpenRouter API"
        except Exception as e:
            print(f"[API EXCEPTION] Error type: {type(e).__name__}")
            print(f"[API EXCEPTION] Error message: {str(e)}")
            import traceback
            print(f"[API EXCEPTION] Traceback:\n{traceback.format_exc()}")
            answer = f"[Error] Failed to get response from {model_config['name']}. Error: {str(e)}"
    # Format the response with model info
    response = f"{answer}\n\n<div style='background: #f0f9ff; border-left: 3px solid #0ea5e9; padding: 8px 12px; margin-top: 10px; border-radius: 4px;'><small style='color: #0369a1; font-weight: 500;'>{model_config['icon']} Answered by {model_config['name']}</small></div>"
    
    # Format model info
    model_info = f"""
<div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <span style="font-size: 2em; margin-right: 10px;">{model_config['icon']}</span>
        <h3 style="margin: 0; color: #2c3e50;">{model_config['name']}</h3>
    </div>
    <p style="color: #5a6c7d; margin: 5px 0;">Optimal model selected for your query</p>
</div>
"""
    
    # Format savings information with a more minimal design
    savings_info = f"""
<div style="background: #ffffff; border: 1px solid #e1e8ed; border-radius: 12px; padding: 20px;">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <p style="color: #8795a1; margin: 0; font-size: 0.9em;">Energy Efficiency</p>
            <p style="color: #22c55e; font-size: 1.5em; font-weight: bold; margin: 5px 0;">
                {savings['energy_saved_percent']:.0f}% saved
            </p>
            <p style="color: #5a6c7d; font-size: 0.85em; margin: 0;">
                {savings['energy_saved']:.6f} kWh reduction
            </p>
        </div>
        <div>
            <p style="color: #8795a1; margin: 0; font-size: 0.9em;">Cost Optimization</p>
            <p style="color: #3b82f6; font-size: 1.5em; font-weight: bold; margin: 5px 0;">
                {savings['cost_saved_percent']:.0f}% saved
            </p>
            <p style="color: #5a6c7d; font-size: 0.85em; margin: 0;">
                ${savings['cost_saved']:.6f} reduction
            </p>
        </div>
    </div>
    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e1e8ed;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="color: #8795a1; margin: 0; font-size: 0.85em;">Environmental Impact</p>
                <p style="color: #5a6c7d; margin: 5px 0;">
                    <span style="color: #10b981; font-weight: bold;">{savings['co2_saved_grams']:.1f}g</span> CO₂ prevented
                </p>
            </div>
            <div style="text-align: right;">
                <p style="color: #8795a1; margin: 0; font-size: 0.85em;">Tokens Processed</p>
                <p style="color: #5a6c7d; margin: 5px 0; font-weight: bold;">{savings['tokens']:,}</p>
            </div>
        </div>
    </div>
</div>
"""
    
    # Add to routing history
    router.routing_history.append({
        "timestamp": time.time(),
        "prompt": message,
        "model": selected_model,
        "savings": savings
    })
    
    print(f"[PROCESS] Response formatted, returning to UI")
    print(f"{'='*60}\n")
    
    return response, model_info, savings_info

def get_statistics() -> str:
    """Get cumulative statistics from routing history"""
    if not router.routing_history:
        return """
<div style="background: #f8fafc; border-radius: 12px; padding: 30px; text-align: center; color: #64748b;">
    <p style="margin: 0;">No queries processed yet</p>
    <p style="margin: 10px 0 0 0; font-size: 0.9em;">Start a conversation to see your impact metrics</p>
</div>
"""
    
    total_queries = len(router.routing_history)
    
    # Placeholder company-wide data (will be replaced with Supabase data later)
    company_total_energy_saved = 12.4567  # kWh
    company_total_cost_saved = 234.89     # USD
    company_total_co2_saved = 4982.7      # grams
    company_total_queries = 8742
    
    stats = f"""
<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 25px;">
    <div style="text-align: center; margin-bottom: 25px;">
        <p style="color: #475569; font-size: 1.1em; margin: 0; font-weight: 500;">{total_queries} queries processed</p>
        <p style="color: #64748b; font-size: 0.9em; margin: 10px 0 0 0;">Model used is shown with each response</p>
    </div>
    
    <div style="border-top: 1px solid #e2e8f0; padding-top: 20px;">
        <h4 style="color: #1e293b; font-size: 1em; margin: 0 0 15px 0; font-weight: 600; text-align: center;">Company Total Saved</h4>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 15px;">
            <div style="background: #f0fdf4; border-radius: 8px; padding: 12px; text-align: center;">
                <p style="color: #166534; font-size: 0.8em; margin: 0;">Energy</p>
                <p style="color: #15803d; font-size: 1.3em; font-weight: bold; margin: 3px 0;">
                    {company_total_energy_saved:.2f}
                </p>
                <p style="color: #166534; font-size: 0.7em; margin: 0;">kWh</p>
            </div>
            
            <div style="background: #eff6ff; border-radius: 8px; padding: 12px; text-align: center;">
                <p style="color: #1e40af; font-size: 0.8em; margin: 0;">Cost</p>
                <p style="color: #2563eb; font-size: 1.3em; font-weight: bold; margin: 3px 0;">
                    ${company_total_cost_saved:.2f}
                </p>
                <p style="color: #1e40af; font-size: 0.7em; margin: 0;">USD</p>
            </div>
        </div>
        
        <div style="background: #f8fafc; border-radius: 8px; padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="text-align: center; flex: 1;">
                    <p style="color: #64748b; font-size: 0.8em; margin: 0;">CO₂ Prevented</p>
                    <p style="color: #0f172a; font-size: 1.1em; font-weight: 600; margin: 3px 0;">
                        {company_total_co2_saved:.1f}g
                    </p>
                </div>
                <div style="text-align: center; flex: 1;">
                    <p style="color: #64748b; font-size: 0.8em; margin: 0;">Total Queries</p>
                    <p style="color: #0f172a; font-size: 1.1em; font-weight: 600; margin: 3px 0;">
                        {company_total_queries:,}
                    </p>
                </div>
            </div>
        </div>
        
        <p style="color: #94a3b8; font-size: 0.75em; text-align: center; margin: 10px 0 0 0; font-style: italic;">
            * Company-wide statistics across all users
        </p>
    </div>
</div>
"""
    
    return stats

# Custom CSS for a more professional look
custom_css = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
}
.message {
    padding: 12px 16px !important;
    border-radius: 8px !important;
}
"""

# Create Gradio interface
with gr.Blocks(
    title="AI Router - Intelligent Model Selection", 
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="gray",
        font=["Inter", "system-ui", "sans-serif"]
    ),
    css=custom_css
) as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("""
            <div style="margin-bottom: 30px;">
                <h1 style="margin: 0; font-size: 2em; font-weight: 600; color: #0f172a;">
                    AI Model Router
                </h1>
                <p style="margin: 10px 0 0 0; color: #64748b; font-size: 1.1em;">
                    Automatically selects between small and large language models based on your query
                </p>
            </div>
            """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                container=True,
                elem_classes=["chat-container"]
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=9,
                    container=False,
                    elem_classes=["message-input"]
                )
                submit = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    min_width=100
                )
        
        with gr.Column(scale=2):
            # Model selection display
            model_display = gr.HTML(
                value="""
                <div style="background: #f8fafc; border-radius: 12px; padding: 20px; text-align: center; color: #64748b;">
                    <p style="margin: 0;">Model selection will appear here</p>
                </div>
                """,
                label="Selected Model"
            )
            
            # Savings metrics
            savings_display = gr.HTML(
                value="""
                <div style="background: #f8fafc; border-radius: 12px; padding: 20px; text-align: center; color: #64748b;">
                    <p style="margin: 0;">Efficiency metrics will appear here</p>
                </div>
                """,
                label="Efficiency Metrics"
            )
            
            # Cumulative stats
            stats_display = gr.HTML(
                value=get_statistics(),
                label="Session Overview"
            )
    
    # Footer with minimal info
    with gr.Row():
        gr.Markdown("""
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 0.85em;">
            <p style="margin: 5px 0;">Comparing small vs large model efficiency • Real-time tracking • Environmental impact monitoring</p>
        </div>
        """)
    
    def respond(message, chat_history):
        response, model_info, savings = process_message(message, chat_history)
        chat_history.append((message, response))
        return "", chat_history, model_info, savings, get_statistics()
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot, model_display, savings_display, stats_display])
    submit.click(respond, [msg, chatbot], [msg, chatbot, model_display, savings_display, stats_display])
    
    # Clear button functionality
    def clear_chat():
        return None, """
        <div style="background: #f8fafc; border-radius: 12px; padding: 20px; text-align: center; color: #64748b;">
            <p style="margin: 0;">Model selection will appear here</p>
        </div>
        """, """
        <div style="background: #f8fafc; border-radius: 12px; padding: 20px; text-align: center; color: #64748b;">
            <p style="margin: 0;">Efficiency metrics will appear here</p>
        </div>
        """, get_statistics()
    
    # Add clear functionality to the Enter key
    msg.submit(lambda: "", outputs=[msg])

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"         AI MODEL ROUTER - STARTUP")
    print(f"{'='*60}")
    print(f"[LAUNCH] Starting Gradio app...")
    print(f"[LAUNCH] Environment: TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')}")
    print(f"[LAUNCH] Models configured:")
    for k, v in MODEL_CONFIGS.items():
        print(f"         - {k}: {v['name']} ({v['icon']})")
    print(f"[LAUNCH] OpenRouter API Key: {'✓ SET' if OPENROUTER_API_KEY else '✗ NOT SET (Demo Mode)'}")
    print(f"{'='*60}\n")
    demo.launch(share=False)