import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import joblib

stocks = ['^DJI', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
dataframes = []
for stock in stocks:
    df = yf.download(stock, start='2020-01-01', end='2025-08-30')
    df['percent_change_price'] = df['Close'].pct_change() * 100
    df['percent_change_next_weeks_price'] = df['percent_change_price'].shift(-1)
    df = df.dropna()
    df['volume'] = df['Volume']
    df['close'] = df['Close']
    df['date'] = df.index
    df['stock_encoded'] = stock
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)

scaler = MinMaxScaler()
df['close'] = scaler.fit_transform(df[['close']])
df['volume'] = scaler.fit_transform(df[['volume']])
df['price_range'] = df['close'].rolling(window=2).max() - df['close'].rolling(window=2).min()
df['volume_ratio'] = df['volume'] / df['volume'].shift(1).replace(0, 1)
df['volume_log'] = np.log1p(df['volume'])
df['volatility'] = df['close'].rolling(window=3).std().fillna(0)
df['ma_3_close'] = df['close'].rolling(window=3).mean().fillna(0)
df['ema_3_close'] = df['close'].ewm(span=3).mean()
df['lag_percent_change_price'] = df['percent_change_price'].shift(1).fillna(0)
df['lag_2_percent_change_price'] = df['percent_change_price'].shift(2).fillna(0)
df['lag_volatility'] = df['volatility'].shift(1).fillna(0)
df['lag_volume_ratio'] = df['volume_ratio'].shift(1).fillna(1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['stock_encoded'] = le.fit_transform(df['stock_encoded'])

joblib.dump(le, 'label_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')

features = ['stock_encoded', 'close', 'volume_log', 'percent_change_price',
            'price_range', 'volume_ratio', 'volatility',
            'lag_percent_change_price', 'lag_2_percent_change_price', 'ma_3_close', 'ema_3_close', 'lag_volatility', 'lag_volume_ratio']
X = df[features].values
y = (df['percent_change_next_weeks_price'] > 0).astype(int).values

def create_sequences(X, y, timesteps=5):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps-1])
    return np.array(Xs), np.array(ys)

timesteps = 5
X_seq, y_seq = create_sequences(X, y, timesteps)

print(f"X shape after sequencing: {X_seq.shape}, y shape: {y_seq.shape}")

class_weights = compute_class_weight('balanced', classes=np.unique(y_seq), y=y_seq)
class_weight_dict = dict(enumerate(class_weights))

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = LSTM(100, return_sequences=False, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, class_weight=class_weight_dict, callbacks=[early_stopping], verbose=1)

model.save('lstm_model.h5')
joblib.dump(model, 'lstm_model.joblib')

y_pred = (model.predict(X_test) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
print(f'LSTM - Accuracy: {acc:.3f}')
print(f'LSTM - Classification Report:\n{report}')

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Down', 'Up']).plot()
plt.title('Confusion Matrix (LSTM)')
plt.show()