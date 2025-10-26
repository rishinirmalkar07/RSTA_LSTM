import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.regularizers import l1 # type: ignore
import tensorflow as tf

class SpatiotemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units, lasso_lambda=0.01):
        super(SpatiotemporalAttention, self).__init__()
        self.units = units
        self.lasso_lambda = lasso_lambda

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 regularizer=l1(self.lasso_lambda))
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        self.V = self.add_weight(shape=(self.units, 1),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        projection = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_scores = tf.tensordot(projection, self.V, axes=1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = inputs * tf.broadcast_to(attention_weights, tf.shape(inputs))
        return tf.reduce_sum(context_vector, axis=1)

def create_rsta_lstm(input_shape, lstm_units, attention_units, lasso_lambda=0.01):
    inputs = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = Dropout(0.2)(x) 
    x = LSTM(lstm_units, return_sequences=True)(x)  
    attention_out = SpatiotemporalAttention(attention_units, lasso_lambda)(x)
    outputs = Dense(1)(attention_out)
    return Model(inputs=inputs, outputs=outputs)

df = pd.read_csv("synthetic_nox_emission_data.csv")

features = df.drop(columns=['NOx_Emission'])
target = df['NOx_Emission']

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

def create_sequences(data, target, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

TIME_STEPS = 10
X, y = create_sequences(scaled_features, target.values, TIME_STEPS)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, shuffle=False)

lstm_units = 64
attention_units = 32
lasso_lambda = 0.01

model = create_rsta_lstm((TIME_STEPS, X.shape[2]), lstm_units, attention_units, lasso_lambda)
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)


y_pred = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(y_test, label='Actual NOx', color='blue', linewidth=1)
plt.plot(y_pred, label='Predicted NOx', color='orange', linestyle='--', linewidth=1)
plt.title("Actual vs Predicted NOx Emissions")
plt.xlabel("Time Step")
plt.ylabel("NOx Emission")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
