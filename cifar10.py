import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import mlflow
import os
import matplotlib.pyplot as plt

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
image_dir = './images'
model_path = './saved_model'


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))
    return model


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    mlflow.set_experiment('Experiment-2')
    with mlflow.start_run(run_name='BaseModel'):

        model = create_model()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        mlflow.tensorflow.autolog(every_n_iter=1, log_models=True, disable=False)
        history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                            validation_data=(x_test, y_test))

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig('./images/accuracy_loss.jpg')

        mlflow.log_artifacts(image_dir, 'images')
        mlflow.log_artifacts(model_path, 'model')
        mlflow.log_param("epoch", 10)
        mlflow.log_param('hidden_layers', 8)
        mlflow.log_param("batch_size", 64)
        for a in history.history['accuracy']:
            mlflow.log_metric("Accuracy", a)
        for ll in history.history['loss']:
            mlflow.log_metric("loss", ll)
        for va in history.history['val_accuracy']:
            mlflow.log_metric("ValidationAccuracy", va)
        for vl in history.history['val_loss']:
            mlflow.log_metric("ValidationLoss", vl)


