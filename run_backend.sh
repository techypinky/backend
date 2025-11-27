#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn codeassistx_mvp_advanced:app --reload --port 8000
