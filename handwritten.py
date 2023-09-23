import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

# Load the dataset
data = pd.read_csv("datasetlink")

# Extract features and labels
X = data.values[:, 1:] 
y = data.values[:, 0]  

# Reshape (28x28x1) features and normalize
X = X.reshape(-1, 28, 28, 1) 
X = X.astype('float32') / 255.0  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
model.save('handwritten_character_model.h5')

# Visualize the training history
print("\nAccuracy and Losses Graphs:")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Visualize some predictions
plt.figure(figsize=(15, 8))
for i in range(30): 
    plt.subplot(5, 6, i+1) 
    index = np.random.randint(0, len(X_test))
    predicted_label = np.argmax(model.predict(X_test[index:index+1]))
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {chr(65 + predicted_label)}')
    plt.axis('off') 

plt.tight_layout() 
plt.show()

# Load the model after training
model = tf.keras.models.load_model('handwritten_character_model.h5') 

# Function to preprocess user-input image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  
    img = img.resize((28, 28))  
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)  
    img = img.astype('float32') / 255.0  
    return img

# Get user input for image file
image_path = input("\nEnter the path of the image: ")

# Preprocess the image
input_image = preprocess_image(image_path)

# Predict the character
predicted_label = np.argmax(model.predict(input_image))

# Show the input image and prediction
plt.figure(figsize=(5, 5))
plt.imshow(input_image.reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {chr(65 + predicted_label)}')
plt.axis('off') 
plt.show()

print(f"The predicted character is: {chr(65 + predicted_label)}") 
