import tensorflow as tf
import numpy as np
import functools

print("--Get data--")
file_data = open("./heart.csv").read().lower()
np.set_printoptions(precision=3, suppress=True)

#print(file_data)

data_rows = file_data.split("\n")
column_names = data_rows[0].split(",")
column_names.pop(0)
data = []
value = []
maxs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i_ in range(1, len(data_rows)):
	i = i_-1
	data.append(data_rows[i_].split(","))
	for j in range(len(data[i])):
		data[i][j] = data[i][j].strip()
	value.append(data[i][-1])
	data[i].pop(0)
	data[i].pop()
	data[i][0] = float(data[i][0])
	data[i][1] = float(data[i][1])
	data[i][2] = float(data[i][2])
	data[i][3] = float(data[i][3])
	if(data[i][4] == "present"):
		data[i][4] = 1.0
	else:
		data[i][4] = 0.0
	data[i][5] = float(data[i][5])
	data[i][6] = float(data[i][6])
	data[i][7] = float(data[i][7])
	data[i][8] = float(data[i][8])
	for j in range(len(data[i])):
		if data[i][j] > maxs[j]:
			maxs[j] = data[i][j]

for i in range(len(data)):
	for j in range(len(data[i])):
		data[i][j] = data[i][j]/maxs[j]

'''for i in range(len(data)):
	for j in range(len(data[i])):
		if not isinstance(data[i][j], float):
			print(i, j)'''

#462 values
split = int(462*0.9)

x_train = np.array(data[:split], dtype=float)
y_train = np.array(value[:split], dtype=float)
x_test = np.array(data[split:], dtype=float)
y_test = np.array(value[split:], dtype=float)

#print(x_train)

print("--Process data--")
#x_train, x_test = x_train / 255.0, x_test / 255.0

print("--Make model--")
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(9,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=5, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

x = np.array(data, dtype=float)
y = np.array(value, dtype=float)

prediction = model.predict(x)
over = 0
under = 0
for i in range(len(prediction)):
	predicted_label = np.argmax(prediction[i])
	if predicted_label != y[i]:
		if y[i] == 1:
			over = over+1
		else:
			under = under+1
print(over, under)
