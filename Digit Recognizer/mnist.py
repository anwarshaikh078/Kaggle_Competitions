from keras.models import load_model
import pandas as pd

model = load_model('HRD_using_cnn_opencv.h5')


predictions = model.predict_classes(test, verbose=1)
pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),
              "Label":predictions}).to_csv("KAGGLE_SUBMISSION_FILE",
                                           index=False,
                                           header=True)