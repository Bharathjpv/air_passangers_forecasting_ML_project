from arima.utils import load_model
from arima.constants import *
from arima import logger
import os

class Arima_Predict:

    @staticmethod
    def predict(date1, date2 ):
        num = (date1.year - date2.year) * 12 + date1.month - date2.month
        model = load_model(os.path.join(ROOT, ARTIFACT_DIR, MODEL_PATH))
        output = model.predict(num)
        logger.info('model prediction done')

        return output

    