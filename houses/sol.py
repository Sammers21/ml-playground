from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt

houses_train = pd.read_csv('train.csv')
print(houses_train.head())
print(houses_train.shape)
X = houses_train.copy()
Y = X.pop('SalePrice')
X = X[['LotArea', 'YearBuilt', 'TotRmsAbvGrd']]
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[3]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mean_absolute_error',
    metrics=['mean_absolute_error']
)
z = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16,
)
plt.plot(z.history['loss'], label='loss')