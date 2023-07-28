import streamlit as st
from PIL import Image
import keras
import numpy as np
import tensorflow as tf
import cv2
from keras.optimizers import Adam
import keras.backend as K
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def iou2D(ytrue, ypred):
    smoothing_factor=0.1
    #y_true_f=K.flatten(y_true)
    #y_pred_f=K.flatten(y_pred)
    intersection = K.sum(ytrue*ypred)
    combined_area = K.sum(ytrue+ypred)
    union_area = combined_area - intersection
    iou = (intersection+smoothing_factor)/(union_area+smoothing_factor)
    return iou

def dice_coef2D(ytrue, ypred):
    smoothing_factor=0.1
    ytrue_f = K.flatten(ytrue)
    ypred_f = K.flatten(ypred)
    intersection = K.sum(ytrue*ypred)
    ytrue_area = K.sum(ytrue)
    ypred_area = K.sum(ypred)
    combined_area = ytrue_area + ypred_area
    dice = 2*((intersection+smoothing_factor)/(combined_area+smoothing_factor))
    return dice
attentionUNet2D_w_path = "C:/Users/Admin/Desktop/DATN_20222/FinalProject/model_weights/model_unet_attention.h5"
unet2D_w_path = "../models_weights/Unet_2D_model.h5"

attUnet_2D_model = keras.models.load_model(attentionUNet2D_w_path, compile=False)
attUnet_2D_model.compile(optimizer=Adam(), loss="binary_crossentropy",
             metrics=["accuracy", iou2D, dice_coef2D,keras.metrics.Precision()])
ImgHeight = 256
ImgWidth = 256
threshold = 0.5
def _drawMask(image, cnts, fill=True):
  image = np.array(image)
  markers = np.zeros((image.shape[0], image.shape[1]))
  heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
  t = 2
  if fill:
    t = -1
  cv2.drawContours(markers, cnts, -1, (255, 0, 0), t)
  mask = markers>0
  image[mask,:] = heatmap_img[mask,:]
  return image

def create_mask_and_binary__image(model, pil_img):

    image= cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(image, (ImgHeight, ImgWidth))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred = model.predict(img)
    pred = np.squeeze(pred)

    msk = Image.fromarray(np.uint8(pred * 255))
    bin_img = Image.fromarray(np.uint8((pred > threshold)*255))

    mask = np.array(bin_img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _bbAndMask(image, cnts)
    _heatmap(image, cnts)
    return msk, bin_img

def _bbAndMask(image, cnts):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  ax1.axis('off')
  ax2.axis('off')
  _bbox(image, cnts, ax1)
  _maskOutline(image, cnts, ax2)
  st.pyplot(fig)

def _bbox(image, cnts, ax):
  ax.imshow(image)
  for c in cnts:
    area = cv2.contourArea(c)
    if area < 10:
      continue
    [x, y, w, h] = cv2.boundingRect(c)
    ax.add_patch(Rectangle((x, y), w, h, color = "red", fill = False))

def _maskOutline(image, cnts, ax):
  img = _drawMask(image, cnts, False)
  ax.imshow(img)

def _drawMask(image, cnts, fill=True):
  image = np.array(image)
  markers = np.zeros((image.shape[0], image.shape[1]))
  heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
  t = 2
  if fill:
    t = -1
  cv2.drawContours(markers, cnts, -1, (255, 0, 0), t)
  mask = markers>0
  image[mask,:] = heatmap_img[mask,:]
  return image

def _heatmap(image, cnts):
  fig2 = plt.figure()
  plt.axis('off')
  hm = st.slider("slider for heatmap", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
  img = _drawMask(image, cnts)
  plt.imshow(img, alpha=hm)
  plt.imshow(image, alpha=1-hm)
  plt.title("heatmap")
  st.pyplot(fig2)

def main():
    st.title("MRI Brain Segmentation")

    button = st.button('press here')

    if button:
        st.write("you pressed")

    option = st.selectbox("Choose a option", ["Original Unet 2D", "Attention Unet 2D", "Unet Transformer"])
    st.write("You choosen:", option)


    uploaded_file = st.file_uploader("Upload Image file", type=["jpg", "jpeg", "png", "tif"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="The uploaded image ", use_column_width=True)
        msk, bin = create_mask_and_binary__image(attUnet_2D_model, image)
        st.image(msk, caption="The mask image of the uploaded Image", use_column_width=True)
        st.image(bin, caption=f"The binary image of the uploaded Image with threshold ={threshold}", use_column_width=True)


if __name__ == '__main__':
    main()

