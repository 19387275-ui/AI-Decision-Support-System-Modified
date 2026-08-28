@echo off
python -m src.preprocess
python -m src.case_base
python -m src.evaluation
streamlit run src/app.py
