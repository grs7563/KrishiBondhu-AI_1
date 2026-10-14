import io, json, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "krishibondhu_paddy.keras"
CLASSES_PATH = HERE.parent / "models" / "class_names.json"
KB_PATH = HERE / "brri_kb.json"
MAP_PATH = HERE / "class_mapping.json"
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "8000000"))
BRRI_URL = "https://riceprofile.brri.gov.bd/cultivation/disease"
app = FastAPI(title="KrishiBondhu AI API", version="1.0.0")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Export the model from Colab first.")
model = tf.keras.models.load_model(MODEL_PATH)
classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))

def get_guidance(label, low):
    if low:
        return {"status_bn":"নিশ্চিত ফলাফল পাওয়া যায়নি",
          "guidance_bn":["মডেলের স্কোর কম; রোগ নির্ধারণের জন্য এই ফলাফলের ওপর নির্ভর করবেন না।",
          "পাতার দুই পাশ ও পুরো গাছের পরিষ্কার ছবি তুলুন।",
          "স্থানীয় কৃষি সম্প্রসারণ কর্মকর্তা/উদ্ভিদ রোগ বিশেষজ্ঞের পরামর্শ নিন।"],
          "sources":[{"title":"BRRI Rice Profile — রোগবালাই","url":BRRI_URL}]}
    if label.lower() == "normal":
        return {"status_bn":"সম্ভাব্য স্বাভাবিক পাতা",
          "guidance_bn":["এটি AI-এর অনুমান; নিয়মিত জমি পর্যবেক্ষণ করুন।",
          "নতুন লক্ষণ দেখা দিলে ছবি তুলে বিশেষজ্ঞের পরামর্শ নিন।",
          "সুষম সার ও সঠিক পানি ব্যবস্থাপনা বজায় রাখুন।"],
          "sources":[{"title":"BRRI Rice Profile — রোগবালাই","url":BRRI_URL}]}
    key = mapping.get(label.lower())
    if not key or key not in kb:
        return {"status_bn":"আপলোড করা BRRI নথিতে নিশ্চিত মিল নেই",
          "guidance_bn":[f"মডেলের শ্রেণি: {label}; BRRI factsheet-এর সঙ্গে নির্ভরযোগ্য মিল নেই।",
          "এটি অন্য রোগ, পোকা বা পুষ্টির সমস্যাও হতে পারে।",
          "রাসায়নিক বা মাত্রা অনুমান করে ব্যবহার করবেন না; কৃষি কর্মকর্তার পরামর্শ নিন।"],
          "sources":[{"title":"BRRI Rice Profile — রোগবালাই","url":BRRI_URL}]}
    item = kb[key]
    return {"status_bn":"সম্ভাব্য মিল — বিশেষজ্ঞের মাধ্যমে নিশ্চিত করুন",
      "disease_bn":key,"disease_english":item.get("english",""),
      "symptoms_bn":item.get("symptoms_bn",[]),
      "risk_factors_bn":item.get("risk_factors_bn",[]),
      "management_before_bn":item.get("management_before_bn",[]),
      "management_after_bn":item.get("management_after_bn",[]),
      "safety_note_bn":"AI ফলাফল প্রাথমিক স্ক্রিনিং মাত্র। রাসায়নিক ব্যবহারের আগে বর্তমান DAE/BRRI নির্দেশনা ও পণ্যের লেবেল যাচাই করুন।",
      "sources":[{"title":item.get("source_file","BRRI factsheet"),"pages":item.get("source_pages","মূল নথি যাচাই করুন"),"url":BRRI_URL}]}

@app.get("/api/health")
def health():
    return {"status":"ok","model_loaded":True,"classes":len(classes)}

@app.get("/api/classes")
def get_classes():
    return {"classes":classes}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg","image/png","image/webp"):
        raise HTTPException(status_code=415, detail="Upload JPG, PNG, or WEBP.")
    data = await file.read()
    if not data: raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_BYTES: raise HTTPException(status_code=413, detail="Image exceeds upload limit.")
    try: image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB").resize((224,224))
    except (UnidentifiedImageError, OSError): raise HTTPException(status_code=400, detail="Unreadable image.")
    arr = np.asarray(image,dtype=np.float32)[None,...]
    scores = model.predict(arr,verbose=0)[0]
    ids = np.argsort(scores)[::-1][:3]
    top = [{"class":classes[int(i)],"score":float(scores[int(i)])} for i in ids]
    best = top[0]
    return {"predicted_class":best["class"],"model_score":best["score"],
      "low_confidence":bool(best["score"] < THRESHOLD),"top_predictions":top,
      "farmer_guidance":get_guidance(best["class"],best["score"] < THRESHOLD)}
