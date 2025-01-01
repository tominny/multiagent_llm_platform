"""
openai_utils.py - This module encapsulates the logic to call GPT-4 
using a multi-agent AutoGen approach for creating and refining USMLE vignettes,
with real-time conversation display in Streamlit.
"""

import os
import json
import streamlit as st
from typing import Tuple
import autogen
from datetime import datetime

# ----- AutoGen Setup -----

# LLM configuration with at least one GPT-4 model
config_list = [
    {
        "model": "gpt-4",
        "api_key": st.secrets["OPENAI_API_KEY"],
        "temperature": 1.0,
    },
]

llm_config = {
    "config_list": config_list,
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

# Agents with real-time display
user_proxy = StreamlitUserProxyAgent(
    name="User_proxy",
    system_message=(
        "Manager: Coordinate the creation and improvement of USMLE STEP 1 clinical vignettes. "
        "Your role is to:\n"
        "1. Have the Vignette-Maker create an initial draft based on the topic\n"
        "2. Then have the Neuro-Evaluator check neurological accuracy\n"
        "3. Have the Vignette-Evaluator assess NBME standards compliance\n"
        "4. Have the Labeler classify the content\n"
        "5. Finally, have the Show-Vignette present the improved version\n"
        "Ensure each agent contributes their expertise, consensus is reached, and suggestions are incorporated into the final version."
    ),
    code_execution_config=code_execution_config,
    human_input_mode="NEVER",
)

vigmaker = StreamlitAssistantAgent(
    name="Vignette-Maker",
    system_message=(
        "You are responsible for creating and refining clinical vignettes for USMLE STEP 1. "
        "When you receive a topic:\n"
        "1. Create an initial draft of a clinically accurate vignette\n"
        "2. Include a stem, lead-in question, and 5 answer choices\n"
        "3. Wait for feedback from other experts before making revisions\n"
        "Format the output exactly as follows:\n"
        "{\n"
        "   'question': ['string'],\n"
        "   'correct_answer': ['string'],\n"
        "   'incorrect_answers': ['string'],\n"
        "   'rationales': ['string'],\n"
        "   'usmle_content_outline': ['string'],\n"
        "}"
    ),
    llm_config=llm_config,
)

evaluator = StreamlitAssistantAgent(
    name="Vignette-Evaluator",
    system_message=(
        "As a NBME standards expert, your role is to:\n"
        "1. Evaluate if the vignette follows NBME item writing style guidelines\n"
        "2. Check if distractors are plausible and educational\n"
        "3. Verify that the question tests appropriate clinical reasoning\n"
        "Provide specific feedback for any violations of NBME standards."
    ),
    llm_config=llm_config,
)

neuro_boss = StreamlitAssistantAgent(
    name="Neuro-Evaluator",
    system_message=(
        "As a neurology expert, evaluate:\n"
        "1. Anatomical accuracy of the case\n"
        "2. Correlation between symptoms and proposed lesion locations\n"
        "3. Accuracy of the laterality of the symptoms and lesion location\n"
        "4. Accuracy of neurological exam findings\n"
        "Provide detailed feedback on any neurological inconsistencies."
    ),
    llm_config=llm_config,
)

labeler = GPTAssistantAgent(
    name="Vignette-Labeler",
    instructions="Properly classify the vignette according to the NBME content outline.",
    llm_config={
        "config_list": config_list,
        "assistant_id": 'asst_PG85C3BIwewAbVuR10iu8Ob6',
    }
)

show_off = StreamlitAssistantAgent(
    name="Show-Vignette",
    system_message=(
        "Your role is to present the final revised vignette after all improvements have been made."
    ),
    llm_config=llm_config,
)

# Set up GroupChat
groupchat = autogen.GroupChat(
    agents=[user_proxy, vigmaker, neuro_boss, evaluator, labeler, show_off],
    messages=[],
    max_round=15,
    speaker_selection_method="auto",
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
            "2. Neuro-Evaluator: Check neurological accuracy\n"
            "3. Vignette-Evaluator: Assess NBME standards compliance\n"
            "4. Vignette-Labeler: Classify the content\n"
            "5. Show-Vignette: Present the final improved version\n\n"
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
