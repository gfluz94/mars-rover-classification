import pandas as pd
import numpy as np

from preprocessing import TRAIN_FINAL_PATH, VAL_FINAL_PATH, TEST_FINAL_PATH, get_and_preprocess_data
from model import get_train_and_val_generators, get_prediction_generator, build_inception_net
from evaluation import evaluate, binary_evaluation_plot


def train(model, train_generator, validation_generator, model_checkpoint, total_epochs=30, fine_tuning=True):
    epochs = total_epochs
    if fine_tuning:
        epochs = (2*total_epochs)//3
    history = model.fit(
            train_generator,
            validation_data=validation_generator,
            callbacks=[model_checkpoint],
            epochs=epochs,
            verbose=1)

    if fine_tuning:
        for layer in model.layers:
            layer.trainable = True

        history = model.fit(
                    train_generator,
                    validation_data=validation_generator,
                    callbacks=[model_checkpoint],
                    initial_epoch=epochs,
                    epochs=total_epochs-epochs,
                    verbose=1)
    
    return model

def save_final_output(test_generator, final_predictions, pred_to_label):
    output = {
        "ImageID": [],
        "label": []
    }
    for img, pred in zip(test_generator.filenames, final_predictions.reshape(-1)):
        img_id = img.split("/")[-1].split(".")[0]
        output["ImageID"].append(int(img_id))
        output["label"].append(pred_to_label[1*(pred>0.5)])
    output = pd.DataFrame(output)
    output.sort_values(by="ImageID").to_csv("./output/submission.csv", index=False)
    print("Output file succesfully saved in `output/submission.csv`")


if __name__=="__main__":
    get_and_preprocess_data()
    train_gen, val_gen, id_to_label = get_train_and_val_generators(TRAIN_FINAL_PATH, VAL_FINAL_PATH, batch_size=100)
    model, model_checkpoint = build_inception_net(neurons_top_dense_layer=1024, learning_rate=0.0001)

    model = train(model=model,
                  train_generator=train_gen,
                  validation_generator=val_gen,
                  model_checkpoint=model_checkpoint,
                  total_epochs=30,
                  fine_tuning=True)

    val_gen = get_prediction_generator(VAL_FINAL_PATH)
    y_proba = model.predict(val_gen)
    y_pred = 1*(y_proba>0.5)
    print(">> Evaluating model on validation set:\n")
    print(evaluate(val_gen.classes, y_pred, y_proba))
    binary_evaluation_plot(val_gen.classes, y_proba)

    test_gen = get_prediction_generator(TEST_FINAL_PATH)
    y_proba = model.predict(test_gen)
    y_pred = 1*(y_proba>0.5)
    save_final_output(test_gen, y_pred, id_to_label)
