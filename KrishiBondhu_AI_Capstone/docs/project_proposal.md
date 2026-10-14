# Project Proposal: KrishiBondhu AI
## Problem statement
Bangladesh rice farmers cultivate Boro, Aus and Aman paddy. Leaf symptoms may be hard to distinguish early, and expert guidance may not be immediately accessible. This project builds a prototype to classify paddy-leaf photos into supported Kaggle Paddy Doctor classes and show Bangla, source-linked guidance from uploaded BRRI factsheets.

## Proposed solution
A farmer uploads a leaf photo to a Streamlit frontend. FastAPI validates and preprocesses the image, runs a trained image classifier and returns top class scores. A conservative mapping connects supported classes to BRRI knowledge summaries. Low-confidence or unsupported classes trigger a warning and referral to local agricultural staff.

## AI/ML approach
Transfer learning with ImageNet-pretrained MobileNetV2, 224×224 RGB inputs, a classification head, stratified train/validation/test split, early stopping and model checkpointing. Evaluate accuracy, balanced accuracy, macro-F1, per-class precision/recall and confusion matrix. Softmax values are model scores, not calibrated probabilities.

## Technology stack
Python, TensorFlow/Keras, MobileNetV2, scikit-learn, FastAPI, Streamlit, Pillow, Git/GitHub and Render.

## Data and safety
Kaggle Paddy Doctor data is subject to competition rules. BRRI factsheets are the management-information source. Not all model classes match BRRI documents. Expert review and independent Bangladesh field validation are required; the app does not automatically prescribe chemical doses.

## Deliverables
Notebook, saved model, frontend, backend API, README, GitHub history, deployed URLs, screenshots and final report.
