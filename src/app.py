import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# تحميل النموذج المدرب
model = joblib.load('src/trained_model.pkl')

# عنوان التطبيق
st.title('Mental Health Prediction')

# جمع المدخلات من المستخدم

gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
profession = st.selectbox("Profession", ['Student', 'Civil Engineer', 'Architect', 'UX/UI Designer',
'Digital Marketer', 'Content Writer', 'Educational Consultant',
'Teacher', 'Manager', 'Chef', 'Doctor', 'Lawyer', 'Entrepreneur',
'Pharmacist'])
academic_pressure = st.slider("Academic Pressure", 0, 10, 5)
work_pressure = st.slider("Work Pressure", 0, 10, 5)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)
study_satisfaction = st.slider("Study Satisfaction", 0, 10, 5)
job_satisfaction = st.slider("Job Satisfaction", 0, 10, 5)
sleep_duration = st.selectbox("Sleep Duration (hours)", ['Less than 5 hours', '5-7 hours', 'More than 7 hours'])
dietary_habits = st.selectbox("Dietary Habits", ['Healthy', 'Unhealthy'])
degree = st.selectbox("Degree", ['Undergraduate', 'Postgraduate'])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ['Yes', 'No'])
work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=8)
financial_stress = st.slider("Financial Stress", 0, 10, 5)
family_history = st.selectbox("Family History of Mental Illness", ['Yes', 'No'])


# تحويل البيانات المدخلة إلى DataFrame
# اسم الكولوم شمال اسم الفاريبل يمين
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Profession': [profession],
    'Academic Pressure': [academic_pressure],
    'Work Pressure': [work_pressure],
    'CGPA': [cgpa],
    'Study Satisfaction': [study_satisfaction],
    'Job Satisfaction': [job_satisfaction],
    'Sleep Duration': [sleep_duration],
    'Dietary Habits': [dietary_habits],
    'Degree': [degree],
    'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
    'Work/Study Hours': [work_study_hours],
    'Financial Stress': [financial_stress],
    'Family History of Mental Illness': [family_history]
})

le=LabelEncoder()
input_data['Gender']=le.fit_transform(input_data['Gender'])
input_data['Profession'] = le.fit_transform(input_data['Profession'])
input_data['Family History of Mental Illness']=le.fit_transform(input_data['Family History of Mental Illness'])
input_data['Dietary Habits']=le.fit_transform(input_data['Dietary Habits'])
input_data['Sleep Duration']=le.fit_transform(input_data['Sleep Duration'])
input_data['Degree']=le.fit_transform(input_data['Degree'])
input_data['Family History of Mental Illness']=le.fit_transform(input_data['Family History of Mental Illness'])
input_data['Have you ever had suicidal thoughts ?']=le.fit_transform(input_data['Have you ever had suicidal thoughts ?'])



prediction_mapping = {0: "No", 1: "Yes"}




# إجراء التنبؤ باستخدام النموذج المدرب
prediction = model.predict(input_data)
predicted_result = prediction_mapping[prediction[0]]


# عرض التنبؤ للمستخدم



st.write(f"HAVE DEPRESSION: {predicted_result}")
