# Multi-Agent USMLE Vignette Creation Platform

This Streamlit application allows users to:

1. **Sign up** and **log in**.
2. **Generate** USMLE vignettes using a multi-agent AutoGen system:
   - Vignette-Maker
   - Neuro-Evaluator
   - Vignette-Evaluator
   - Vignette-Labeler
   - Show-Vignette
3. **View** and **store** each vignette’s topic, initial draft, final version, and entire conversation JSON in a SQLite database.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
