import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_save_model():
    # تحميل البيانات
    df = pd.read_csv('workspace/pp.csv')
    df = df.drop(columns=['id'], errors='ignore')  # 'errors="ignore"' يعني أنه إذا كان العمود غير موجود، لا تظهر أي خطأ
    df=df.drop(columns=['City'],errors='ignore')
    # حذف عمود "id" إذا كان موجودًا
    
    # تقسيم البيانات
    X = df.drop(columns=['Depression'])
    y = df['Depression']
    
    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # إنشاء النموذج وتدريبه
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # التنبؤ وقياس الدقة
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    
    # حفظ النموذج المدرب
    joblib.dump(model, 'src/trained_model.pkl')

# تدريب وحفظ النموذج
train_and_save_model()
