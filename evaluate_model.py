import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# dataset path
validation_dir = "dataset/validation"

# load model
model = tf.keras.models.load_model("model/ms_model.h5")

# image generator
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# predictions
predictions = model.predict(generator)

y_pred = np.argmax(predictions, axis=1)
y_true = generator.classes

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix")
print(cm)

# classification report
print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=generator.class_indices.keys()))

# plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=generator.class_indices.keys(),
            yticklabels=generator.class_indices.keys())

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()