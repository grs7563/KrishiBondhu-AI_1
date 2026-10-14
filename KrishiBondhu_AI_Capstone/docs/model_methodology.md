# Model methodology
- Input: RGB image resized to 224×224.
- Backbone: ImageNet-pretrained MobileNetV2 with classification head.
- Data: Kaggle Paddy Doctor labelled training images; notebook uses stratified 70/15/15 train/validation/test split.
- Metrics: accuracy, balanced accuracy, macro-F1, per-class precision/recall and confusion matrix.
- Inference: top three softmax scores. The threshold is heuristic and is not a calibrated confidence guarantee.
- Advisory: conservative rule-based mapping to uploaded BRRI factsheet summaries; no generative model invents treatments.
- Limitation: Kaggle labels and BRRI topics do not perfectly overlap; unsupported labels receive generic safety guidance.
