import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Laddar bilder från dataset-mappen, ändrar storlek till 224x224 och delar upp dem i tränings- och valideringsdata
dataset = image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

train_dataset, val_dataset = dataset

# Klassnamn
class_names = train_dataset.class_names
print("Klasser:", class_names)

# Förbättrar prestandan vid datainläsning
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Normalisering och dataaugmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Laddar en förtränad modell utan toppskikt
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Fryser först basmodellen
base_model.trainable = False

# Bygger modellen
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# Kompilerar modellen
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Tränar modellen
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

# Sparar modellen
model.save("plant_disease_model.keras")

print("Modellen har sparats som plant_disease_model.keras")