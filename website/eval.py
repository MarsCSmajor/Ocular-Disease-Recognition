

def model_image_prediction(model_path,image_path):
    from tensorflow import keras
    import tensorflow as tf
    import numpy as np

    classes = ['Normal', 'Retinal Vascular', 'Optic Nerve', 'Lens', 'Macular/Retinal', 'Other']

    model = keras.models.load_model(model_path)

    image = tf.keras.utils.load_img(image_path,color_mode = 'rgb').resize((48,48)) # resize it

    image_to_array = tf.keras.utils.img_to_array(image)
    image_to_array = image_to_array / 255 # normalize itto improve the constrast in an image

    flat_image = image_to_array.flatten()

    if flat_image.shape[0] > 2082: # because the model a shape of size 2082, anything bigger, the model will not accept it. we resize the length vector of the image
        flat_image = flat_image[:2082]
    
    else:
        flat_image = np.pad(flat_image,(0,2082 - flat_image[0])) # pads at the end of flat_image array with zeros
        # if an image is under 2082, we zero out until the desired length of 2082 is reached to feed it on the model
    
    input_vector = np.expand_dims(flat_image,axis=0) # batch dimensions

    predict = model.predict(input_vector)

    class_predict = np.argmax(predict[0]) # this will give you a number 

    return classes[class_predict]


