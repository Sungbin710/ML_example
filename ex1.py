import pandas as pd
import tensorflow as tf

레몬에이드 = pd.read_csv('lemonade.csv')
print(레몬에이드)

독립 = 레몬에이드[['온도']]
종속 = 레몬에이드[['판매량']]
print(독립.shape, 종속.shape)

X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')

model.fit(독립, 종속, epochs=1000, verbose=0)   #verbose: Verbosity mode => 0 = silent, 1 = progress bar, 2 = one line per epoch

print("Predictions: ", model.predict([15]))
