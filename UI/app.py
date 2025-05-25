import gradio as gr
import json
import time
from typing import Dict, Tuple, List
import random
from bertmodel import predict_label
import tiktoken
from ecologits import EcoLogits
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()
OPENAI_API_KEY = ""
OPENROUTER_API_KEY = ""
# Model configurations with energy consumption and cost estimates
MODEL_CONFIGS = {
    "gpt-4": {
        "name": "GPT-4",
        "energy_per_token": 0.0001,  # kWh per token (estimated)
        "cost_per_token": 0.00003,   # USD per token
        "capabilities": ["complex_reasoning", "coding", "creative_writing", "analysis"],
        "performance": 10,
        "icon": "🧠"
    },
    "gpt-3.5": {
        "name": "GPT-3.5",
        "energy_per_token": 0.00005,
        "cost_per_token": 0.000002,
        "capabilities": ["general_qa", "simple_coding", "summarization"],
        "performance": 7,
        "icon": "💡"
    },
    "claude-instant": {
        "name": "Claude Instant",
        "energy_per_token": 0.00004,
        "cost_per_token": 0.0000008,
        "capabilities": ["quick_responses", "basic_analysis", "chat"],
        "performance": 6,
        "icon": "⚡"
    },
    "llama-7b": {
        "name": "Llama 7B",
        "energy_per_token": 0.00002,
        "cost_per_token": 0.0000002,
        "capabilities": ["basic_qa", "simple_tasks"],
        "performance": 4,
        "icon": "🦙"
    },
    "Mistral-small3.1": {
        "name": "Mistral-small3.1",
        "energy_per_token": 0.00001,
        "cost_per_token": 0.0000003,
        "capabilities": ["factual_queries", "current_events", "specific_data"],
        "performance": 3,
        "icon": "🔍"
    }
}

class ModelRouter:
    def __init__(self):
        self.routing_history = []
    
    def classify_prompt(self, prompt: str) -> str:
        return predict_label(prompt)
    
    def select_model(self, prompt: str) -> str:
        """Select the most efficient model based on prompt classification."""
        prompt_type = self.classify_prompt(prompt)
        # 1) Normalize
        key = prompt_type.strip().lower()

        # 2) Map normalized labels to actual MODEL_CONFIGS keys
        model_map = {
            "search engine":       "claude-instant",  
            "reasoning model":     "gpt-4",
            "developer model":     "gpt-3.5",
            "large language model":"gpt-4",
            "small language model":"llama-7b",
        }

        # 3) Never KeyError: use .get() with a sensible default
        selected = model_map.get(key, "gpt-4")  
        return selected
    

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

        if response is not None:
            response_tokens = len(encoding.encode(response))
        elif max_response_tokens is not None:
            # you’re reserving this many tokens for the model’s reply
            response_tokens = max_response_tokens
        else:
            # fallback to a default budget
            response_tokens = 0

        return prompt_tokens + response_tokens
    
    def gpt4o_get_energy(self, prompt: str) -> float:
        """
        Calculates baseline energy consumption for a given prompt using gpt-4o-mini via EcoLogits.
        NOTE: Ensure EcoLogits is initialized appropriately for your application.
            Calling EcoLogits.init() on every function call might not be optimal.
            Consider initializing it once globally if appropriate for the library's design.
        """
        try:
            # Initialize EcoLogits - Consider if this needs to be done globally once
            EcoLogits.init()
            
            client = OpenAI(
                api_key= OPENAI_API_KEY
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt }
                ],
                # Ensure that 'impacts' are requested if EcoLogits requires specific parameters
                # For example, some wrappers might need extra_body={"impacts": True} or similar
            )
            # Accessing impacts can vary based on the EcoLogits version and OpenAI library version.
            # The following assumes 'response.impacts.energy.value' is the correct path.
            # If EcoLogits patches the response object directly.
            if hasattr(response, 'impacts') and response.impacts and hasattr(response.impacts, 'energy') and response.impacts.energy:
                energy_consumption = response.impacts.energy.value
                #ghg_emission = response.impacts.gwp.value
                return energy_consumption
            else:
                print("Warning: EcoLogits impact data not found in response. Returning 0 for baseline.")
                # Fallback if impact data is not available as expected
                # You might want to raise an error or handle this differently
                return 0
        except Exception as e:
            print(f"Error in baseline_energy function: {e}")
            # Fallback or re-raise error
            return 0 # Return a default or handle error appropriately
    
    def calculate_savings(self, selected_model: str, prompt: str) -> Dict:
        """Calculate energy and cost savings compared to using GPT-4"""
        tokens = self.estimate_tokens(prompt)
        
        selected_config = MODEL_CONFIGS[selected_model]
        gpt4o_config = MODEL_CONFIGS["gpt-4"]
        
        # Calculate actual usage
        actual_energy = tokens * selected_config["energy_per_token"]
        actual_cost = tokens * selected_config["cost_per_token"]
        
        # Calculate GPT-4 usage
        gpt4o_energy_temp = self.gpt4o_get_energy(prompt)
        gpt4o_energy = (gpt4o_energy_temp.min + gpt4o_energy_temp.max) / 2
        gpt4o_cost = tokens * gpt4o_config["cost_per_token"]
        
        # Calculate savings
        energy_saved = gpt4o_energy - actual_energy
        cost_saved = gpt4o_cost - actual_cost
        energy_saved_percent = (energy_saved / gpt4o_energy) * 100 if gpt4o_energy > 0 else 0
        cost_saved_percent = (cost_saved / gpt4o_cost) * 100 if gpt4o_cost > 0 else 0
        
        return {
            "selected_model": selected_config["name"],
            "tokens": tokens,
            "actual_energy": actual_energy,
            "actual_cost": actual_cost,
            "gpt4o_energy": gpt4o_energy,
            "gpt4o_cost": gpt4o_cost,
            "energy_saved": energy_saved,
            "cost_saved": cost_saved,
            "energy_saved_percent": energy_saved_percent,
            "cost_saved_percent": cost_saved_percent,
            "co2_saved_grams": energy_saved * 400  # Approximate CO2 per kWh
        }

router = ModelRouter()

def process_message(message: str, history: List[List[str]]) -> Tuple[str, str, str]:
    """Process the user message and return response with savings info"""
    
    # Route to appropriate model
    selected_model = router.select_model(message)
    model_config = MODEL_CONFIGS[selected_model]
    
    # Calculate savings
    savings = router.calculate_savings(selected_model, message)
    
    open_router_model_dict = {
        "gpt-4": "openai/gpt-4o",
        "llama-7b": "alfredpros/codellama-7b-instruct-solidity",
    }
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    },
    data=json.dumps({
        "model": open_router_model_dict.get(selected_model, "openai/gpt-4o"), # Optional
        "messages": [
        {
            "role": "user",
            "content": message
        }
        ]
    })
    )   
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    # Simulate model response (in real implementation, this would call actual APIs)
    if selected_model == "search_engine":
        response = f"Based on search results: [This would be real search results for: {message}]\n\n<div style='background: #f0f9ff; border-left: 3px solid #0ea5e9; padding: 8px 12px; margin-top: 10px; border-radius: 4px;'><small style='color: #0369a1; font-weight: 500;'>{model_config['icon']} Answered by {model_config['name']}</small></div>"
    else:
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
                    AI Efficiency Router
                </h1>
                <p style="margin: 10px 0 0 0; color: #64748b; font-size: 1.1em;">
                    Intelligent model selection for optimal performance and sustainability
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
            <p style="margin: 5px 0;">Comparing against GPT-4 baseline • Real-time efficiency tracking • Environmental impact monitoring</p>
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
    demo.launch(share=False)