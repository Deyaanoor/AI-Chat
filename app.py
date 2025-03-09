from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyngrok import ngrok
import os

# تعطيل GPU مؤقتًا
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# تعيين التوكن الصحيح لـ ngrok
ngrok.set_auth_token("2u57yIppZ1YmD5ExbkdjHLVEyuE_5rk1FjpJdbzbaHhs3FL8o")

app = Flask(__name__)
CORS(app)

# تحميل النموذج والبيانات
print("جارٍ تحميل النموذج والبيانات...")
model = tf.keras.models.load_model('car_problems_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("تم تحميل النموذج والبيانات بنجاح!")

def open_ngrok_tunnel():
    try:
        # إغلاق أي أنفاق سابقة
        ngrok.kill()
        time.sleep(1)  # انتظار 1 ثانية للتأكد من الإغلاق
        
        # فتح النفق الجديد مع إعدادات خاصة
        public_url = ngrok.connect(5001, bind_tls=True)
        print(f" * تم فتح النفق بنجاح: {public_url}")
        return public_url
    except Exception as e:
        print(f" * فشل في فتح النفق: {str(e)}")
        return None

# تشغيل ngrok
print("جارٍ فتح نفق ngrok...")
public_url = open_ngrok_tunnel()

if public_url:
    print(f" * الرابط العام للتطبيق: {public_url}")
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            
            # التحقق من وجود جميع الحقول
            required_fields = ['Make', 'Model', 'Problem', 'Symptoms', 'Year']
            if not all(field in data for field in required_fields):
                return jsonify({"error": "معلومات ناقصة في الطلب"}), 400

            # تجهيز البيانات
            text_input = f"{data['Make']} {data['Model']} {data['Problem']} {data['Symptoms']}"
            sequences = tokenizer.texts_to_sequences([text_input])
            padded_sequences = pad_sequences(sequences, maxlen=50)
            
            # إضافة السنة كخاصية رقمية
            input_data = np.hstack([padded_sequences, np.array([[data['Year']]])])
            
            # التنبؤ
            prediction = model.predict(input_data)
            predicted_solution = label_encoder.inverse_transform([np.argmax(prediction)])
            
            return jsonify({"Solution": predicted_solution[0]}), 200

        except Exception as e:
            return jsonify({"error": f"خطأ في المعالجة: {str(e)}"}), 500

    if __name__ == '__main__':
        print("جارٍ تشغيل الخادم...")
        app.run(host='0.0.0.0', port=5001, debug=False)  # تعطيل وضع التصحيح

else:
    print(" ! فشل في تشغيل الخادم بسبب مشاكل في ngrok")
