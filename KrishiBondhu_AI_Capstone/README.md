# KrishiBondhu AI
AI-based paddy-leaf disease screening and Bangla farmer advisory for Bangladesh.

## Problem statement
Paddy farmers cultivate rice during Boro, Aus and Aman seasons. Symptoms can be difficult to distinguish and expert advice may not be immediately available. This prototype classifies leaf photos into supported Kaggle Paddy Doctor categories and presents relevant Bangla guidance summarized from the uploaded BRRI factsheets. It is a screening aid, not a definitive diagnosis.

## Dataset and source
- Kaggle: https://www.kaggle.com/competitions/paddy-disease-classification/data
- BRRI: https://riceprofile.brri.gov.bd/cultivation/disease
Dataset access/use is subject to Kaggle competition rules. Confirm permission before redistribution.

## Architecture
Streamlit frontend → FastAPI API → image validation and resize → MobileNetV2 classifier → conservative BRRI mapping → Bangla result with source link.

## Local setup
Use Python 3.10/3.11 for best TensorFlow compatibility. The model must exist at `backend/models/krishibondhu_paddy.keras`.
Backend terminal:
```bash
cd backend
python -m venv .venv
# activate venv, then:
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
API docs: Open KrishiBondhu AI API Documentation

Frontend in another terminal:
```bash
cd frontend
python -m venv .venv
# activate venv, then:
pip install -r requirements.txt
# Windows PowerShell:
$env:KRISHIBONDHU_API_URL="http://127.0.0.1:8000"
# macOS/Linux:
export KRISHIBONDHU_API_URL="http://127.0.0.1:8000"
streamlit run app.py
```

## Evaluation
Report actual test accuracy, balanced accuracy, macro-F1, per-class precision/recall and confusion matrix from the notebook. Softmax scores are not calibrated probabilities. A random Kaggle split is not independent Bangladesh field validation.

## Render deployment
Create two web services from this repo.
Backend: Root Directory `backend`; Build `pip install -r requirements.txt`; Start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
Frontend: Root Directory `frontend`; Build `pip install -r requirements.txt`; Start `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
Set frontend environment variable `KRISHIBONDHU_API_URL` to the deployed backend URL. Set backend `ALLOWED_ORIGINS` to frontend URL. Test backend `/api/health` and upload a real photo. Free tier limits can change.

## GitHub
```bash
git init
git add README.md backend frontend docs .gitignore render.yaml
git commit -m "Create KrishiBondhu AI capstone"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/KrishiBondhu-AI.git
git push -u origin main
```
Never commit `kaggle.json`, API keys, private farm images, or the dataset. Use Git LFS or model artifact hosting if the model is too large.

## Limitations
Only classes represented in the training labels can be predicted. Some Kaggle labels do not match uploaded BRRI factsheets and receive a safe fallback. Advice must be reviewed by BRRI/DAE or a plant pathologist. No automatic pesticide-dose prescription is provided.
