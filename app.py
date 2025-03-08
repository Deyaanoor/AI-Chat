from flask import Flask, request, jsonify
from flask_cors import CORS  # إضافة هذه السطر
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# تحميل النموذج
model = tf.keras.models.load_model('car_problems_model.h5')

# تحميل tokenizer و label_encoder
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# تعريف التطبيق
app = Flask(__name__)
CORS(app)  # تمكين CORS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # الحصول على المدخلات من الطلب بتنسيق JSON
        data = request.get_json()

        # التأكد من أن البيانات المدخلة تحتوي على كل المدخلات المطلوبة
        make = data.get('Make')
        model_input = data.get('Model')
        problem = data.get('Problem')
        symptoms = data.get('Symptoms')
        year = data.get('Year')

        # التأكد من أن جميع المدخلات موجودة
        if not make or not model_input or not problem or not symptoms or not year:
            return jsonify({"error": "Missing input data"}), 400

        # دمج المدخلات النصية كما في عملية التحضير
        text_input = f"{make} {model_input} {problem} {symptoms}"

        # تحويل النص المدخل إلى تسلسل من الأرقام باستخدام tokenizer
        sequences = tokenizer.texts_to_sequences([text_input])
        padded_sequences = pad_sequences(sequences, maxlen=50)

        # إضافة السنة كميزة
        input_data = np.hstack([padded_sequences, np.array([[year]])])

        # التنبؤ باستخدام النموذج
        prediction = model.predict(input_data)

        # فك تشفير النتيجة إلى النصوص الأصلية باستخدام label_encoder
        predicted_solution = label_encoder.inverse_transform([np.argmax(prediction)])

        # تحضير النتيجة للإرجاع (بدون Difficulty)
        result = {
            "Solution": predicted_solution[0]
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
