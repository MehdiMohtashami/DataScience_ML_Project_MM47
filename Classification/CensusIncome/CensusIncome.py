import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# مسیر فایل‌ها و نام ستون‌ها
# -----------------------
train_path = 'adult.data.csv'
test_path  = 'adult.test.csv'

cols = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# ------------ خواندن فایل‌ها ------------
# از names استفاده می‌کنیم تا ستون‌ها قطعاً نام‌گذاری شوند.
# skipinitialspace کمک می‌کند تا فاصله‌های اضافی بعد از ویرگول حذف شود.
train_df = pd.read_csv(train_path, header=None, names=cols, na_values='?', skipinitialspace=True)
test_df  = pd.read_csv(test_path,  header=None, names=cols, na_values='?', skipinitialspace=True, comment='|')

# در بعضی نسخه‌های adult.test ممکن است سطر اول شامل عناوین یا مقادیر اضافی باشد.
# پاکسازی اولیه: حذف ردیف‌هایی که در ستون‌ها به‌طور اشتباه حاوی نام ستون هستند
for c in cols:
    # برابر بودن رشته‌ای با نام ستون (حساسیت به حروف کوچک/بزرگ را کم می‌کنیم)
    mask = test_df[c].astype(str).str.strip().str.lower() == c.lower()
    if mask.any():
        test_df = test_df[~mask]

# ترکیب موقت برای پردازش یکسان
full_df = pd.concat([train_df, test_df], ignore_index=True)

# حذف ردیف‌هایی که کاملاً خالی هستند یا همه مقادیرشان NaN
full_df = full_df.dropna(how='all').reset_index(drop=True)

# برخی ردیف‌های خراب ممکن است حاوی متن 'age' یا سایر headerها در هر ستونی باشند.
# برای اطمینان: در هر ستون اگر مقدار دقیقا برابر با نام ستون بود، آن ردیف را حذف کن.
for c in cols:
    full_df = full_df[full_df[c].astype(str).str.strip().str.lower() != c.lower()]

full_df = full_df.reset_index(drop=True)

# ------------- تبدیل ستون‌های عددی به عدد و حذف ردیف‌های خراب -------------
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# سعی می‌کنیم همه numeric_cols را به عدد تبدیل کنیم؛ مقادیر غیرقابل‌تبدیل => NaN
for nc in numeric_cols:
    full_df[nc] = pd.to_numeric(full_df[nc], errors='coerce')

# اگر مایل باشی می‌توانی NaN‌ها را ایمپوت کنی؛ اینجا ساده‌ترین راه را می‌ریم: حذف ردیف‌هایی که در ستون‌های عددی NaN دارند
before = len(full_df)
full_df = full_df.dropna(subset=numeric_cols).reset_index(drop=True)
after = len(full_df)
print(f"Dropped {before - after} rows because numeric columns contained invalid values.")

# برای ستون‌های دسته‌ای، پر کردن NaN با مد (در صورت وجود)
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country', 'income']
for c in categorical_cols:
    if full_df[c].isna().sum() > 0:
        full_df[c] = full_df[c].fillna(full_df[c].mode()[0])

# ----------------- انکُدینگ: یک LabelEncoder برای هر ستون دسته‌ای -----------------
encoders = {}
from sklearn.preprocessing import LabelEncoder
for col in ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country']:
    le = LabelEncoder()
    full_df[col] = full_df[col].astype(str)
    le.fit(full_df[col])
    full_df[col] = le.transform(full_df[col])
    encoders[col] = le

# income را هم انکُد کن
income_le = LabelEncoder()
full_df['income'] = income_le.fit_transform(full_df['income'].astype(str))
encoders['income'] = income_le

# -------------- برگرداندن به train/test با طول‌های اصلی --------------
train_len = sum(1 for _ in open(train_path))  # تعداد ردیف‌های اصلی فایل آموزش (شامل هر header غیرمعمول)
# به‌جای محاسبه‌ی طول فایل، بهتر طول اولیه train_df را نگه داریم؛ چون ممکنه خطوط خراب حذف شده باشند.
# اگر می‌خواهی دقیق‌تر: از طولِ read شده‌ی train_df قبل از concat استفاده کن که ما آن را داریم:
# اما چون قبلاً train_df خوانده شد، از آن استفاده می‌کنیم:
train_rows = train_df.shape[0]
# حالا با توجه به اینکه برخی ردیف‌ها از test_df حذف شده‌اند، بهتر تقسیم طبق طول train_df اولیه انجام شود:
train_processed = full_df.iloc[:train_rows].reset_index(drop=True)
test_processed  = full_df.iloc[train_rows:].reset_index(drop=True)

# --------- آماده‌سازی ماتریس ویژگی‌ها و هدف ---------
feature_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

# اطمینان از اینکه همه ستون‌ها وجود دارند
missing = [c for c in feature_names if c not in train_processed.columns]
if missing:
    raise ValueError("Missing features in processed dataframe: " + ", ".join(missing))

X_train = train_processed[feature_names].astype(float)
y_train = train_processed['income'].astype(int)
X_test  = test_processed[feature_names].astype(float)
y_test  = test_processed['income'].astype(int)

# ---------- استانداردسازی ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------- آموزش مدل ----------
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)

# ---------- ارزیابی ----------
y_pred = model.predict(X_test_scaled)
print("Accuracy on test:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------- ذخیره‌ها ----------
joblib.dump(model, 'best_xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("Saved: best_xgboost_model.pkl, scaler.pkl, encoders.pkl, feature_names.pkl")
