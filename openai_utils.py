"""
openai_utils.py - This module encapsulates the logic to call GPT-4 
using a multi-agent AutoGen approach for creating and refining USMLE vignettes,
with real-time conversation display in Streamlit.
"""

import os
import json
import streamlit as st
from typing import Tuple, Optional, Dict, Any
import autogen
from datetime import datetime
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

# ----- AutoGen Setup -----

# LLM configuration with at least one GPT-4 model
config_list = [
    {
        "model": "gpt-4",
        "api_key": st.secrets["OPENAI_API_KEY"],
    },
]

llm_config = {
    "config_list": config_list,
    "temperature": 1.0,
    "cache_seed": None,
    "timeout": 120,
}

# Code execution configuration
code_execution_config = {
    "work_dir": "coding",
    "use_docker": False,
    "last_n_messages": 3,
}

# Initialize session state for message storage
if 'messages' not in st.session_state:
    st.session_state.messages = []

def update_chat_display(sender: str, message: str):
    """Update the Streamlit display with new message"""
    st.session_state.messages.append({"sender": sender, "content": message})
    with st.chat_message(sender):
        st.markdown(message)

# Custom User Proxy Agent with message display
class StreamlitUserProxyAgent(autogen.UserProxyAgent):
    def send(self, message: str, recipient: autogen.Agent, request_reply: bool = True, **kwargs) -> None:
        update_chat_display(self.name, message)
        super().send(message, recipient, request_reply=request_reply)

# Custom Assistant Agent with message display
class StreamlitAssistantAgent(autogen.AssistantAgent):
    def send(self, message: str, recipient: autogen.Agent, request_reply: bool = True, **kwargs) -> None:
        update_chat_display(self.name, message)
        super().send(message, recipient, request_reply=request_reply)

# Custom GPT Assistant Agent with error handling
class SafeGPTAssistantAgent(GPTAssistantAgent):
    def send(self, message: str, recipient: autogen.Agent, request_reply: bool = True, **kwargs) -> None:
        try:
            update_chat_display(self.name, message)
            super().send(message, recipient, request_reply=request_reply, **kwargs)
        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}"
            update_chat_display(self.name, error_msg)
            if request_reply:
                recipient.send("I encountered an error. Please continue with the next step.", self)

# Agents with real-time display
user_proxy = StreamlitUserProxyAgent(
    name="User_proxy",
    system_message=(
        "Manager: Coordinate the creation and improvement of USMLE STEP 1 clinical vignettes. "
        "Your role is to:\n"
        "1. Have the Vignette-Maker create an initial draft based on the topic\n"
        "2. Then have the Content-Checker check medical accuracy\n"
        "3. Then have the Format-Checker assess NBME standards compliance\n"
        "4. Once feedback is received, have the Show-Vignette present the improved final version\n"
        "Overall, ensure each agent contributes their expertise and suggestions are incorporated into the final version.\n"
    ),
    code_execution_config=code_execution_config,
    human_input_mode="NEVER",
)

vignette_maker = StreamlitAssistantAgent(
    name="Vignette-Maker",
    system_message=(
        "You are responsible for creating initial clinical vignettes for USMLE STEP 1. "
        "When you receive a topic:\n"
        "1. Create a draft of a clinically accurate vignette\n"
        "2. Include a stem, lead-in question, and 5 answer choices\n"
        "3. Wait for feedback from other experts\n"
        "Format the output as follows:\n"
        "{\n"
        "   'question': ['string - the vignette text and question'],\n"
        "   'correct_answer': ['string - the correct answer'],\n"
        "   'incorrect_answers': ['string list - the incorrect answers'],\n"
        "   'rationales': ['string - explanation why the correct answer is right'],\n"
        "   'usmle_content_outline': ['string list - relevant content areas'],\n"
        "}"
    ),
    llm_config=llm_config,
)

format_checker = StreamlitAssistantAgent(
    name="Format-Checker",
    system_message=(
        "As a NBME standards expert, evaluate the vignette against this comprehensive checklist:\n\n"

        "**PATIENT PRESENTATION:**\n"
        "□ Age and sex clearly stated\n"
        "□ Chief complaint or presenting symptom included\n"
        "□ Relevant history (duration, onset, progression)\n"
        "□ Pertinent physical exam findings\n"
        "□ Relevant lab/diagnostic results if needed\n"
        "□ Realistic clinical scenario (not contrived)\n\n"

        "**QUESTION STRUCTURE:**\n"
        "□ Lead-in question is clear and specific\n"
        "□ Question stem provides all necessary information\n"
        "□ One clearly correct answer\n"
        "□ Exactly 5 answer choices (A-E)\n"
        "□ Tests clinical reasoning, not just memorization\n"
        "□ Appropriate difficulty level for STEP 1\n\n"

        "**ANSWER CHOICES:**\n"
        "□ All choices are homogeneous (same category/format)\n"
        "□ Choices listed in logical order (alphabetical, anatomical, or numerical)\n"
        "□ NO 'All of the above' or 'None of the above'\n"
        "□ NO combinations (e.g., 'A and B', 'Both 1 and 3')\n"
        "□ Distractors are plausible and represent common errors\n"
        "□ Distractors are educational (not obviously wrong)\n"
        "□ Similar length and grammatical structure across choices\n\n"

        "**WRITING STYLE:**\n"
        "□ AVOID negative stems ('Which is NOT...', 'All EXCEPT...')\n"
        "□ AVOID absolute terms ('always', 'never', 'only', 'must')\n"
        "□ AVOID 'Aunt Minnie' pattern recognition questions\n"
        "□ AVOID leading clues or hints toward correct answer\n"
        "□ Use clear, concise medical language\n"
        "□ Avoid unnecessary information\n\n"

        "**VIOLATIONS TO FLAG:**\n"
        "- Negative phrasing in lead-in\n"
        "- Implausible or 'throw-away' distractors\n"
        "- Missing patient demographics\n"
        "- Questions testing only recall vs. application\n"
        "- Heterogeneous answer choices\n"
        "- Grammatical clues to correct answer\n\n"

        "Provide specific, actionable feedback citing which checklist items are violated or well-executed."
    ),
    llm_config=llm_config,
)

content_checker = StreamlitAssistantAgent(
    name="Content-Checker",
    system_message=(
        "You are an expert medical educator who checks the clinical accuracy of USMLE questions. "
        "Please evaluate the vignette for:\n"
        "1. Clinical plausibility and accuracy\n"
        "2. Correct diagnosis and treatment options\n"
        "3. Appropriate difficulty level for STEP 1\n"
        "Provide specific feedback on any medical inaccuracies."
    ),
    llm_config=llm_config,
)

show_vignette = StreamlitAssistantAgent(
    name="Show-Vignette",
    system_message=(
        "Your role is to present the final revised vignette after all improvements have been made. "
        "Incorporate feedback from Content-Checker and Format-Checker to improve the original vignette. "
        "Present the final version in a clear, formatted way."
    ),
    llm_config=llm_config,
)

# Set up GroupChat with simplified agent list (removed Vignette-Labeler)
groupchat = autogen.GroupChat(
    agents=[user_proxy, vignette_maker, content_checker, format_checker, show_vignette],
    messages=[],
    max_round=12,
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
)

# Manager orchestrates the conversation
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

def generate_usmle_vignette(topic: str) -> Tuple[str, str, str]:
    """
    Generate a USMLE-style clinical vignette via the multi-agent system.
    Shows live conversation in Streamlit.
    """
    try:
        # Clear previous messages at the start of new generation
        st.session_state.messages = []
        
        # Create containers for versions
        initial_container = st.container()
        final_container = st.container()
        
        prompt = (
            f"Let's create a USMLE STEP 1 clinical vignette about {topic}. "
            "Each agent will contribute their expertise:\n\n"
            "1. Vignette-Maker: Start by creating an initial draft\n"
            "2. Content-Checker: Check medical accuracy\n"
            "3. Format-Checker: Assess NBME standards compliance\n"
            "4. Show-Vignette: Present the final improved version\n\n"
            "Vignette-Maker, please begin by creating a vignette about this topic."
        )

        # Start the conversation
        with st.spinner('Initiating conversation between agents...'):
            result = user_proxy.initiate_chat(
                manager,
                message=prompt
            )

        # Process results
        initial_vignette = None
        final_vignette = None
        
        for msg in st.session_state.messages:
            if msg["sender"] == "Vignette-Maker" and not initial_vignette:
                initial_vignette = msg["content"]
                with initial_container:
                    st.info("Initial Draft")
                    st.markdown(msg["content"])
            
            elif msg["sender"] == "Show-Vignette":
                try:
                    final_vignette = msg["content"]
                    with final_container:
                        st.success("Final Version")
                        st.markdown(msg["content"])
                except Exception as e:
                    st.warning(f"Error processing final vignette: {str(e)}")

        # Convert conversation to JSON for storage
        conversation_json = json.dumps(st.session_state.messages, indent=2)

        if not initial_vignette:
            initial_vignette = "No initial vignette found."
        if not final_vignette:
            final_vignette = "No final vignette found."

        return (initial_vignette, final_vignette, conversation_json)

    except Exception as e:
        st.error(f"Error generating vignette: {str(e)}")
        return (str(e), "", json.dumps({"error": str(e)}))

if __name__ == "__main__":
    st.title("USMLE Vignette Generator")
    st.markdown("### Enter a topic to generate a clinical vignette")
    
    topic = st.text_input("Topic:", "memory loss")
    if st.button("Generate Vignette"):
        init_vig, final_vig, convo = generate_usmle_vignette(topic)
