from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold

diseases = [
    {'name': 'ไข้หวัด', 'symptoms': ['ไข้', 'ไอ', 'เจ็บคอ', 'น้ำมูกไหล']},
    {'name': 'ไข้หวัดใหญ่', 'symptoms': ['ไข้สูง', 'ไอ', 'เจ็บคอ', 'ปวดกล้ามเนื้อ', 'อ่อนเพลีย']},
    {'name': 'ไมเกรน', 'symptoms': ['ปวดศีรษะ', 'คลื่นไส้', 'แพ้แสง']},
    {'name': 'โรคเบาหวาน', 'symptoms': ['ปัสสาวะบ่อย', 'กระหายน้ำ', 'น้ำหนักลด', 'อ่อนเพลีย']},
    {'name': 'โรคความดันโลหิตสูง', 'symptoms': ['ปวดศีรษะ', 'ตาพร่ามัว', 'เจ็บหน้าอก', 'หายใจเหนื่อย']},
    {'name': 'หอบหืด', 'symptoms': ['ไอ', 'หายใจมีเสียงหวีด', 'หายใจเหนื่อย', 'แน่นอก']},
    {'name': 'โรคหลอดลมอักเสบ', 'symptoms': ['ไอ', 'มีเสมหะ', 'อ่อนเพลีย', 'หายใจเหนื่อย']},
    {'name': 'ปอดอักเสบ', 'symptoms': ['ไข้', 'ไอ', 'เจ็บหน้าอก', 'หายใจเหนื่อย']},
    {'name': 'วัณโรค', 'symptoms': ['ไอต่อเนื่อง', 'น้ำหนักลด', 'เหงื่อกลางคืน', 'ไข้']},
    {'name': 'COVID-19', 'symptoms': ['ไข้', 'ไอแห้ง', 'อ่อนเพลีย', 'สูญเสียรสชาติหรือกลิ่น']},
    {'name': 'โรคหัวใจ', 'symptoms': ['เจ็บหน้าอก', 'หายใจเหนื่อย', 'ใจสั่น', 'เหงื่อออกมาก']},
    {'name': 'โรคหัวใจ', 'symptoms': ['ใจสั่น']},
    {'name': 'โรคหลอดลมอักเสบ', 'symptoms': ['มีเสมหะ']},
    {'name': 'ปอดอักเสบ', 'symptoms': ['เจ็บหน้าอก']},
    {'name': 'โรคความดันโลหิตสูง', 'symptoms': ['ตาพร่ามัว']},
    {'name': 'โรคเบาหวาน', 'symptoms': ['ปัสสาวะบ่อย']},
    {'name': 'ไข้หวัดใหญ่', 'symptoms': ['ไข้สูง']},
    {'name': 'หอบหืด', 'symptoms': ['หายใจเหนื่อย']},
    {'name': 'วัณโรค', 'symptoms': ['เหงื่อกลางคืน']},
    {'name': 'COVID-19', 'symptoms': ['สูญเสียรสชาติหรือกลิ่น']},
    {'name': 'ไข้หวัดใหญ่', 'symptoms': ['ไข้สูง']},
    {'name': 'ไข้หวัด', 'symptoms': ['ไข้']},
    {'name': 'ไข้หวัด', 'symptoms': ['น้ำมูกไหล']},
    {'name': 'ไมเกรน', 'symptoms': ['ปวดศีรษะ']}
]

df = pd.DataFrame(diseases)

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['symptoms'])
y = df['name']

clf = DecisionTreeClassifier()

kf = KFold(n_splits=3, shuffle=True, random_state=1)
kfold_scores = cross_val_score(clf, X, y, cv=kf)
print("k-Fold Cross-Validation Scores:", kfold_scores)
mean_kfold_score = kfold_scores.mean()
print("Mean k-Fold Cross-Validation Score:", mean_kfold_score)

loo = LeaveOneOut()
loo_scores = cross_val_score(clf, X, y, cv=loo)
print("Leave-One-Out Cross-Validation Scores:", loo_scores)
mean_loo_score = loo_scores.mean()
print("Mean Leave-One-Out Cross-Validation Score:", mean_loo_score)

clf.fit(X, y)

def diagnose_disease(symptoms):
    symptoms_transformed = mlb.transform([symptoms])
    predicted_disease = clf.predict(symptoms_transformed)
    return predicted_disease[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', symptoms=mlb.classes_)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    user_symptoms = request.form.getlist('symptoms')
    predicted_disease = diagnose_disease(user_symptoms)
    
    advice = {
        'ไข้หวัด': 'พักผ่อน ดื่มน้ำให้เพียงพอ และใช้ยารักษาไข้หวัดทั่วไป หากอาการไม่ดีขึ้น ควรพบแพทย์',
        'ไข้หวัดใหญ่': 'พักผ่อน ดื่มน้ำ และรับยาต้านไวรัสถ้ามีการสั่งจ่าย ควรพบแพทย์หากอาการรุนแรง',
        'ไมเกรน': 'พักผ่อนในห้องมืด เงียบ และใช้ยาระงับปวด หากไมเกรนเกิดบ่อย ควรปรึกษาแพทย์',
        'โรคเบาหวาน': 'รักษาอาหารที่สมดุล ออกกำลังกายสม่ำเสมอ และตรวจระดับน้ำตาลในเลือด ปรึกษาผู้เชี่ยวชาญทางการแพทย์',
        'โรคความดันโลหิตสูง': 'ลดการบริโภคเกลือ ออกกำลังกายสม่ำเสมอ และหลีกเลี่ยงความเครียด ปรึกษาแพทย์เพื่อการรักษาระยะยาว',
        'หอบหืด': 'ใช้ยาพ่นตามที่แพทย์สั่ง หลีกเลี่ยงสิ่งกระตุ้น และควบคุมอารมณ์ในช่วงที่มีอาการหอบ ควรพบแพทย์หากหายใจลำบาก',
        'โรคหลอดลมอักเสบ': 'พักผ่อน ดื่มน้ำ และใช้เครื่องทำความชื้นในอากาศ หากอาการแย่ลงควรพบแพทย์',
        'ปอดอักเสบ': 'พักผ่อน รับประทานยาปฏิชีวนะถ้ามีการสั่งจ่าย และดื่มน้ำ ควรพบแพทย์หากหายใจลำบาก',
        'วัณโรค': 'รับประทานยาทุกชนิดที่แพทย์สั่งและปฏิบัติตามคำแนะนำทางการแพทย์ การกักกันอาจจำเป็นเพื่อป้องกันการแพร่กระจาย',
        'โรคหัวใจ': 'พักผ่อน หลีกเลี่ยงความเครียด ควรรับประทานยาตามที่แพทย์สั่ง และพบแพทย์ทันทีหากมีอาการรุนแรง',
        'COVID-19': 'แยกตัวเอง พักผ่อน ดื่มน้ำ และเฝ้าระวังอาการ ควรพบแพทย์หากอาการแย่ลง โดยเฉพาะหากหายใจลำบาก'
    }

    return render_template('result.html', disease=predicted_disease, advice=advice.get(predicted_disease, 'ไม่มีคำแนะนำสำหรับโรคนี้'))

if __name__ == '__main__':
    app.run(debug=True)