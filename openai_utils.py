"""
openai_utils.py - This module encapsulates the logic to call GPT-4 (or ChatGPT4-o)
using a multi-agent AutoGen approach for creating and refining USMLE vignettes.
"""

import os
import json
import streamlit as st
from typing import Tuple
import autogen

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

# Code execution configuration (if needed for code-generation agents)
code_execution_config = {
    "work_dir": "coding",
    "use_docker": False,
    "last_n_messages": 3,
}

# Agents
user_proxy = autogen.UserProxyAgent(
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

vigmaker = autogen.AssistantAgent(
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

evaluator = autogen.AssistantAgent(
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

neuro_boss = autogen.AssistantAgent(
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

labeler = autogen.AssistantAgent(
    name="Vignette-Labeler",
    system_message=(
        "Your role is to properly classify the vignette according to the NBME content outline."
    ),
    llm_config=llm_config,
)

show_off = autogen.AssistantAgent(
    name="Show-Vignette",
    system_message=(
        "Your role is to present the final revised vignette after all improvements have been made."
    ),
    llm_config=llm_config,
)

# Create a GroupChat that includes all agents
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

    Returns a tuple: (initial_vignette, final_vignette, conversation_json)
    1) initial_vignette: The text from Vignette-Maker
    2) final_vignette: The final improved version from Show-Vignette
    3) conversation_json: The entire conversation in JSON
    """
    try:
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

        # Start the multi-agent conversation
        result = user_proxy.initiate_chat(
            manager,
            message=prompt,
            silent=False,  # set to False if you want more debug logs in the terminal
        )

        conversation_history = result.chat_history

        # Convert entire conversation to JSON
        conversation_json = json.dumps(conversation_history, indent=2)

        # 1) Find the initial vignette from Vignette-Maker
        initial_vignette = None
        for msg in conversation_history:
            if msg.get("name") == "Vignette-Maker":
                initial_vignette = msg.get("content", "")
                break

        # 2) Find the final vignette from Show-Vignette
        final_vignette = None
        for msg in reversed(conversation_history):
            if msg.get("name") == "Show-Vignette":
                final_vignette = msg.get("content", "")
                break

        if not initial_vignette:
            initial_vignette = "No initial vignette found."
        if not final_vignette:
            final_vignette = "No final vignette found."

        return (initial_vignette, final_vignette, conversation_json)

    except Exception as e:
        # Return placeholders on error
        error_msg = f"Error generating multi-agent vignette: {str(e)}"
        return (error_msg, "", json.dumps({"error": str(e)}))


if __name__ == "__main__":
    # Quick local test in your console
    TEST_TOPIC = "memory issue"
    init_vig, final_vig, convo = generate_usmle_vignette(TEST_TOPIC)
    print("=== INITIAL VIGNETTE ===\n", init_vig)
    print("\n=== FINAL VIGNETTE ===\n", final_vig)
    print("\n=== FULL CONVERSATION JSON ===\n", convo)
