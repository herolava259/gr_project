from keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D

import keras
from keras.models import Model
def conv_block(inp,filters):
    x=Conv2D(filters,(1,1),padding='same',activation='relu')(inp)
    x=Conv2D(filters,(1,1),padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    return x
def encoder_block(inp,filters,dropout):
    x=conv_block(inp,filters)
    p=MaxPooling2D(pool_size=(2,2))(x)
    p=Dropout(dropout)(p)
    return x,p

def attention_block(l_layer,h_layer): #Attention Block
    phi=Conv2D(h_layer.shape[-1],(1,1),padding='same')(l_layer)
    theta=Conv2D(h_layer.shape[-1],(1,1),strides=(2,2),padding='same')(h_layer)
    x=keras.layers.add([phi,theta])
    x=Activation('relu')(x)
    x=Conv2D(1,(1,1),padding='same',activation='sigmoid')(x)
    x=UpSampling2D(size=(2,2))(x)
    x=keras.layers.multiply([h_layer,x])
    x=BatchNormalization(axis=3)(x)
    return x

def decoder_block(inp,filters,concat_layer,dropout):
    x=Conv2DTranspose(filters,(2,2),strides=(2,2),padding='same')(inp)
    concat_layer=attention_block(inp,concat_layer)
    x=concatenate([x,concat_layer])
    x=Dropout(dropout)(x)
    x=conv_block(x,filters)
    return x

def attention_2D_model(ImgHeight, ImgWidth):
    input_img = Input((ImgHeight, ImgWidth, 3), name='img')
    d1, p1 = encoder_block(input_img, 64, 0.1)
    d2, p2 = encoder_block(p1, 128, 0.1)
    d3, p3 = encoder_block(p2, 256, 0.1)
    d4, p4 = encoder_block(p3, 512, 0.1)
    b1 = conv_block(p4, 1024)
    e2 = decoder_block(b1, 512, d4, 0.1)
    e3 = decoder_block(e2, 256, d3, 0.1)
    e4 = decoder_block(e3, 128, d2, 0.1)
    e5 = decoder_block(e4, 64, d1, 0.1)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(e5)
    model = Model(inputs=[input_img], outputs=[outputs], name='AttentionUnet')

    return model

