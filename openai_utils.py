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
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

# ----- AutoGen Setup -----

# LLM configuration with at least one GPT-4 model
config_list = [
    {
        "model": "gpt-4",
        "api_key": st.secrets["OPENAI_API_KEY"],
        # temperature removed from here
    },
]

llm_config = {
    "config_list": config_list,
    "temperature": 1.0,  # Temperature moved here
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
        "1. Have the Question-Generator create an initial draft based on the topic\n"
        "2. Then have the Content-Checker check neurological accuracy\n"
        "3. Then have the Format-Checker assess NBME standards compliance\n"
        "4. Have the Vignette-Labeler classify the content once\n"
        "5. Have the Question-Generator address the suggestions and present to the team an improved vignette version for further scrutiny.\n"
        "5. Once consensus is reached, have the Show-Vignette present the improved final version\n"
        "Overall, ensure each agent contributes their expertise, errors are addressed and consensus is reached, and suggestions are incorporated into the next vignette version.\n"
        "Iterate among the agents until each agent is satisfied and confims that issues have been addressed."
    ),
    code_execution_config=code_execution_config,
    human_input_mode="NEVER",
)

vignette_maker = StreamlitAssistantAgent(
    name="Vignette-Maker",
    system_message=(
        "You are responsible for creating an initial clinical vignettes for USMLE STEP 1 and refine it based on recommendations form the other agents. "
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

format_checker = StreamlitAssistantAgent(
    name="Format-Checker",
    system_message=(
        "As a NBME standards expert, your role is to:\n"
        "1. Evaluate if the vignette follows NBME item writing style guidelines\n"
        "2. Check if distractors are plausible and educational\n"
        "3. Verify that the question tests appropriate clinical reasoning\n"
        "Provide specific feedback for any violations of NBME standards."
    ),
    llm_config=llm_config,
)

# Updated Content-Checker agent
content_checker = StreamlitAssistantAgent(
    name="Content-Checker",
    system_message=(
        "You are an expert medical educator with extensive knowledge of clinical science & medicine, pathophysiology, and standard-of-care guidelines. "
        "You use chain-of-thought reasoning, self-check logic, differential diagnosis exploration, and self-consistency to thoroughly analyze the factual and scientific accuracy of board-style questions.\n"
        "Please evaluate the vignette question.\n\n"
        "Inside your reasoning (which you will show to the user), do the following:\n"
        "1. Generate multiple lines of reasoning (self-consistency), exploring different ways the vignette could have been framed for the topic. Discuss whether each version is accurate, considering variations in clinical presentation, demographics, or risk factors.\n"
        "2. Consider the key differential diagnoses, referencing guideline-based care if needed. Comment on whether the vignette's details support or contradict each potential diagnosis.\n"
        "3. Perform an internal self-check to ensure the scenario is medically accurate and consistent. Identify any contradictions, missing information, or clinically unlikely details.\n"
        "4. Provide your final evaluation of the USMLE-style question, including feedback on:\n"
        "   - The short clinical vignette (patient age, gender, relevant symptoms, labs/imaging).\n"
        "   - The 5 answer choices (one correct, four distractors).\n"
        "   - Whether the labeled correct answer is indeed correct.\n"
        "   - How well the rationale justifies that choice in a concise 2–3 sentence explanation.\n\n"
        "Important:\n"
        "- In the final feedback you present (for the student or question-writer), do not disclose your chain-of-thought, detailed differential discussions, or self-check logic.\n"
        "- Only provide a concise, synthesized critique describing the strengths, weaknesses, and overall accuracy of the vignette and its answer choices."
    ),
    llm_config=llm_config,
)


# GPT Assistant Agent for labeling - Updated config structure
vignette_labeler = GPTAssistantAgent(
    name="Vignette-Labeler",
    instructions="You are a medical educator. Properly classify the vignette according \n"
    "to the National Board of Examiners (NBME) content outline for USMLE vignette questions that is part of your knowledge base.",
    llm_config={
        "config_list": [
            {
                "model": "gpt-4",
                "api_key": st.secrets["OPENAI_API_KEY"],
                # No temperature here
            }
        ],
        "temperature": 1.0,  # Temperature at top level
        "assistant_id": 'asst_N78lM1DPedMCZTGo6PIgpBe1',
    }
)

show_vignette = StreamlitAssistantAgent(
    name="Show-Vignette",
    system_message=(
        "Your role is to present the final revised vignette after all improvements have been made. Replace the NBME classification with that from the Vignette-Labeler "
    ),
    llm_config=llm_config,
)

# Set up GroupChat
groupchat = autogen.GroupChat(
    agents=[user_proxy, vignette_maker, content_checker, format_checker, vignette_labeler, show_vignette],
    messages=[],
    max_round=15,
    speaker_selection_method="round_robin",  # Changed from "auto" to "round_robin" to avoid potential division issues
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
            "2. Content-Checker: Check neurological accuracy\n"
            "3. Format-Checker: Assess NBME standards compliance\n"
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

Version 2 of 2
