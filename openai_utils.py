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
    agents=[user
