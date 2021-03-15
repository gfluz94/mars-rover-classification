from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


WEIGHTS_URL = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
WEIGHTS_FILE = "./log/00_inception_v3.h5"


def get_train_and_val_generators(train_path, val_path, batch_size=100):
    print("Preparing file streamer...")
    train_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    print("\t", end="")
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        target_size=(256, 256))

    validation_datagen = ImageDataGenerator(rescale=1./255)
    print("\t", end="")
    validation_generator = validation_datagen.flow_from_directory(val_path,
                                                                  batch_size=batch_size,
                                                                  class_mode='binary',
                                                                  target_size=(256, 256))
    print("File streamer is ready!!!\n")
    print("Checking class labels:")
    print(f"\t{train_generator.class_indices}")
    print(f"\t{validation_generator.class_indices}")

    pred_to_label = {v:k for k, v in train_generator.class_indices.items()}

    return train_generator, validation_generator, pred_to_label

def get_prediction_generator(files_path):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(files_path,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      class_mode='binary',
                                                      target_size=(256, 256))
    return test_generator

def build_inception_net(neurons_top_dense_layer=1024, learning_rate=0.0001):
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)

    pre_trained_model = InceptionV3(input_shape=(256, 256, 3),
                                    include_top=False,
                                    weights=None)

    pre_trained_model.load_weights(WEIGHTS_FILE)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    print('Last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    x = Flatten()(last_output)
    x = Dense(neurons_top_dense_layer, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)

    model_checkpoint = ModelCheckpoint(
                            filepath="./log/mars-rover-{epoch:02d}.hdf5",
                            save_best_only=False,
                            save_weights_only=True,
                            monitor='val_accuracy',
                            mode='max'
                        )

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model, model_checkpoint

