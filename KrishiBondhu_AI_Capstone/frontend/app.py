import os, requests, streamlit as st
from PIL import Image
st.set_page_config(page_title="KrishiBondhu AI",page_icon="🌾",layout="centered")
API_URL=os.getenv("KRISHIBONDHU_API_URL","http://127.0.0.1:8000").rstrip("/")
st.title("🌾 KrishiBondhu AI")
st.subheader("ধানের পাতার রোগ শনাক্তকরণ ও বাংলা কৃষি পরামর্শ")
st.write("পাতার পরিষ্কার ছবি আপলোড করুন। এটি প্রাথমিক AI স্ক্রিনিং, নিশ্চিত রোগনির্ণয় নয়।")
uploaded=st.file_uploader("ধানের পাতার ছবি দিন (JPG, PNG, WEBP)",type=["jpg","jpeg","png","webp"])
if uploaded:
    st.image(Image.open(uploaded),caption="আপলোড করা ছবি",use_container_width=True)
    if st.button("রোগের সম্ভাবনা ও পরামর্শ দেখুন",type="primary"):
        try:
            uploaded.seek(0)
            r=requests.post(f"{API_URL}/api/predict",files={"file":(uploaded.name,uploaded.getvalue(),uploaded.type or "image/jpeg")},timeout=120)
            r.raise_for_status(); result=r.json()
        except requests.RequestException as e:
            st.error(f"সার্ভারে যোগাযোগ করা যায়নি। API URL পরীক্ষা করুন। {e}"); st.stop()
        st.markdown("### ফলাফল")
        st.write("**সম্ভাব্য শ্রেণি:**",result["predicted_class"])
        st.write(f"**মডেলের স্কোর:** {result['model_score']*100:.1f}%")
        if result["low_confidence"]: st.warning("স্কোর কম; কৃষি কর্মকর্তার পরামর্শ নিন।")
        g=result["farmer_guidance"]; st.info(g.get("status_bn",""))
        if g.get("disease_english"): st.write("**Disease class:**",g["disease_english"])
        for key,title in [("symptoms_bn","সম্ভাব্য লক্ষণ"),("risk_factors_bn","ঝুঁকির কারণ"),("management_before_bn","প্রতিরোধ ও আগাম ব্যবস্থাপনা"),("management_after_bn","লক্ষণ দেখা দিলে করণীয়"),("guidance_bn","সাধারণ নির্দেশনা")]:
            if g.get(key):
                st.markdown("#### "+title)
                for line in g[key]: st.markdown("- "+line)
        if g.get("safety_note_bn"): st.warning(g["safety_note_bn"])
        st.markdown("#### সূত্র")
        for s in g.get("sources",[]): st.markdown(f"- [{s.get('title','BRRI source')}]({s.get('url','https://riceprofile.brri.gov.bd/cultivation/disease')})")
st.caption("রাসায়নিক ব্যবহারের আগে বর্তমান BRRI/DAE নির্দেশনা ও পণ্যের লেবেল যাচাই করুন।")
