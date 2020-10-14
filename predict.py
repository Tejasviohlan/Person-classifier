import numpy as np
from keras.models import load_model
from keras.preprocessing import image

def family(filename):

    model = load_model('model.h5')

    test_image = image.load_img(filename, target_size = (90, 90))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict(test_image)

    if result[0][0] == 1:
        prediction = 'dam dami'
        return [{ "image" : prediction}]
    elif result[0][1]==1:
        prediction = 'maa'
        return [{ "image" : prediction}]
    elif result[0][2]==1:
        prediction = 'sony'
        return [{ "image" : prediction}]
    else:
        return [{ "image" : "wrong image"}]