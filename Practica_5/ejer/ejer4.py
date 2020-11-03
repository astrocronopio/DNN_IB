# Utilizando los datos y la arquitectura que considere oportuna 
# describir los distintos procesospara observar lo que la red 
# neuronal ha aprendido.

from tensorflow.keras.applications import MobileNet, mobilenet
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
import tensorflow as tf


img_w = 300
img_h = 300
crop_img = 5
model = MobileNet(weights="imagenet", 
                  include_top=False)

layer = model.get_layer(index=-3)
feature_extractor = Model(inputs = model.inputs, 
                          outputs= layer.output)

"""
La siguiente función se la copié a Chollet sin asco,
es la única cosa que no terminé de entender
"""
def compute_loss(input_image, layer_filter):
    activation = feature_extractor(input_image) #
    filter_activation = activation[:, 2:-2, 2:-2, layer_filter]
    return tf.reduce_mean(filter_activation)
"""
"""

def grad_asc(img, layer_filter, lr):
    with tf.GradientTape() as g: #Hasta en los docs está así
        g.watch(img)
        loss = compute_loss(img, layer_filter) # el siguiente gradiente es de esta función
    grads = g.gradient(loss, img) # La magia existe
    img += lr * grads
    return loss, img

def fit_grad_asc(layer_filter, n_epochs):
    lr = 10.0 #si es chico no se ve la psicodelia
    img = init_image()
    loss= 0.0
    for _ in range(n_epochs):
        loss, img = grad_asc(img, layer_filter, lr)

    img = output_image(img[0].numpy()) # Lo paso sin esa dimensión extra, como un np.array
    return loss, img


def init_image(): 
    """
    Imagenes del perro y del triceratops
    """
    #image = load_img("dog.10003.jpg",  target_size=(img_w, img_h, 3))
    #image = load_img("socutteeee.png", target_size=(img_w, img_h, 3))
    
    # input_arr = img_to_array(image) 
    
    # input_arr-=input_arr.mean()
    # input_arr/=input_arr.std()   + 1e-5
    
    # input_arr= 0.2*np.clip(input_arr, -1, 1)
    # img = tf.reshape(input_arr, [1, img_w, img_h, 3]) #No sé porque quiere una dimensión más con el  1
    """
    Imagen del seno
    """     
    # img = 0.2*tf.sin(0.008*(tf.range(0, img_h*img_w*3, dtype=tf.float32)%256))
    # img = tf.reshape(img, [1, img_w, img_h, 3])
    """
    Imagen random
    """
    img = 0.2*(tf.random.uniform((1, img_w, img_h, 3))-0.5)
    return img

def output_image(img):
    #proceso inverso
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.2

    img = img[crop_img: -crop_img, 
              crop_img: -crop_img, :]

    img = np.clip((img + 0.5)*255, 0, 255) # RGB
    
    return img


def plot_filters(all_imgs, n, m, filename, n_epochs):   
    margin = 10
    crop_w = img_w - crop_img * 2
    crop_h = img_h - crop_img * 2
    
    w   = n * crop_w + (n - 1) * margin
    h   = m * crop_h + (m - 1) * margin
    
    filters = np.zeros((w, h, 3))

    for i in range(n):
        for j in range(m):
            img = all_imgs[i * m + j]
            filters[(crop_w + margin) * i : (crop_w + margin) * i + crop_w, #limite x
                    (crop_h + margin) * j : (crop_h + margin) * j + crop_h, #limite y
                    :,] = img
            
    save_img("{}_filters_epochs_{}.pdf".format(filename, n_epochs), filters)
    

def ejer4(filename, n_epochs):
    img_init = init_image()
    img_init = output_image(img_init[0].numpy())    
    save_img("{}_init_img_epochs_{}.pdf".format(filename, n_epochs), img_init)
    
    all_imgs = []
    n, m = 1, 2 # array para el plot

    for layer_filter in range(n*m):
        layer_filter= layer_filter + 2
        loss, img = fit_grad_asc(layer_filter, n_epochs)
        all_imgs.append(img)
    
    plot_filters(all_imgs, n, m, filename, n_epochs)
    
if __name__ == '__main__':
    ejer4("random", 50)
    