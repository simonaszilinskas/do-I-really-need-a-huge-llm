import gradio as gr
import json
import time
from typing import Dict, Tuple, List
import random

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
    "search_engine": {
        "name": "Search Engine",
        "energy_per_token": 0.00001,
        "cost_per_token": 0.0000001,
        "capabilities": ["factual_queries", "current_events", "specific_data"],
        "performance": 3,
        "icon": "🔍"
    }
}

class ModelRouter:
    def __init__(self):
        self.routing_history = []
    
    def classify_prompt(self, prompt: str) -> str:
        """Classify the prompt type based on keywords and complexity"""
        prompt_lower = prompt.lower()
        
        # Check for search engine queries
        search_keywords = ["what is the weather", "current news", "stock price", 
                         "latest", "today", "now", "current", "real-time"]
        if any(keyword in prompt_lower for keyword in search_keywords):
            return "search_query"
        
        # Check for complex reasoning
        complex_keywords = ["explain in detail", "analyze", "compare and contrast",
                          "philosophical", "ethical", "complex", "advanced"]
        if any(keyword in prompt_lower for keyword in complex_keywords):
            return "complex_reasoning"
        
        # Check for coding
        coding_keywords = ["code", "program", "function", "algorithm", "debug",
                         "implement", "python", "javascript", "sql"]
        if any(keyword in prompt_lower for keyword in coding_keywords):
            return "coding"
        
        # Check for creative writing
        creative_keywords = ["write a story", "poem", "creative", "imagine",
                           "fiction", "narrative"]
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return "creative_writing"
        
        # Default to general Q&A
        return "general_qa"
    
    def select_model(self, prompt: str) -> str:
        """Select the most efficient model based on prompt classification"""
        prompt_type = self.classify_prompt(prompt)
        
        # Model selection logic
        model_map = {
            "search_query": "search_engine",
            "complex_reasoning": "gpt-4",
            "coding": "gpt-3.5",
            "creative_writing": "gpt-4",
            "general_qa": "llama-7b"
        }
        
        # For simple questions, we might use an even lighter model
        if len(prompt.split()) < 10 and "?" in prompt:
            return "llama-7b"
        
        return model_map.get(prompt_type, "claude-instant")
    
    def estimate_tokens(self, prompt: str, response_length: str = "medium") -> int:
        """Estimate token count for prompt and response"""
        # Simple estimation: ~1.3 tokens per word
        prompt_tokens = int(len(prompt.split()) * 1.3)
        
        response_multipliers = {
            "short": 50,
            "medium": 150,
            "long": 300
        }
        response_tokens = response_multipliers.get(response_length, 150)
        
        return prompt_tokens + response_tokens
    
    def calculate_savings(self, selected_model: str, prompt: str) -> Dict:
        """Calculate energy and cost savings compared to using GPT-4"""
        tokens = self.estimate_tokens(prompt)
        
        selected_config = MODEL_CONFIGS[selected_model]
        gpt4_config = MODEL_CONFIGS["gpt-4"]
        
        # Calculate actual usage
        actual_energy = tokens * selected_config["energy_per_token"]
        actual_cost = tokens * selected_config["cost_per_token"]
        
        # Calculate GPT-4 usage
        gpt4_energy = tokens * gpt4_config["energy_per_token"]
        gpt4_cost = tokens * gpt4_config["cost_per_token"]
        
        # Calculate savings
        energy_saved = gpt4_energy - actual_energy
        cost_saved = gpt4_cost - actual_cost
        energy_saved_percent = (energy_saved / gpt4_energy) * 100 if gpt4_energy > 0 else 0
        cost_saved_percent = (cost_saved / gpt4_cost) * 100 if gpt4_cost > 0 else 0
        
        return {
            "selected_model": selected_config["name"],
            "tokens": tokens,
            "actual_energy": actual_energy,
            "actual_cost": actual_cost,
            "gpt4_energy": gpt4_energy,
            "gpt4_cost": gpt4_cost,
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
    
    # Simulate model response (in real implementation, this would call actual APIs)
    if selected_model == "search_engine":
        response = f"Based on search results: [This would be real search results for: {message}]\n\n<div style='background: #f0f9ff; border-left: 3px solid #0ea5e9; padding: 8px 12px; margin-top: 10px; border-radius: 4px;'><small style='color: #0369a1; font-weight: 500;'>{model_config['icon']} Answered by {model_config['name']}</small></div>"
    else:
        response = f"[This would be the actual model response to: {message}]\n\n<div style='background: #f0f9ff; border-left: 3px solid #0ea5e9; padding: 8px 12px; margin-top: 10px; border-radius: 4px;'><small style='color: #0369a1; font-weight: 500;'>{model_config['icon']} Answered by {model_config['name']}</small></div>"
    
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
    demo.launch(share=True)