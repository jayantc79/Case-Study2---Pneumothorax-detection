import tensorflow as tf


def final_fun_2(X,Y):
  img = tf.io.read_file(X)
  image = tf.image.decode_png(img)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image=tf.squeeze(image,[0])
  b = tf.constant([1,1,3], tf.int32)
  image=tf.tile(image,b)
  image=tf.image.resize(image,size=[256,256])
  image=tf.expand_dims(image,axis=0)
  if Y!=" -1":
    print("Ground truth of Classification is 1(Has Pneumothorax)")
    print('*'*100)
  else:
    print("Ground truth of Classification is 0(Does not have Pneumothorax)")
    print("Ground truth of Segmentation -- There is no mask")
    print('*'*100)

    
  if model.predict(image)>=0.5:
    print("Pneumothorax has been detected")
    mask=final.predict(image)
    mask=(mask>0.5).astype(np.uint8)
    try:
      true_mask=Image.fromarray(mask_functions.rle2mask(Y,1024,1024).T).resize((256,256), resample=Image.BILINEAR)
      true_mask=np.array(true_mask)
      plt.figure(figsize=(20,6))
      plt.subplot(121)
      plt.title("X-ray image with mask(Ground truth)")
      plt.imshow(np.squeeze(image),cmap='gray')
      plt.imshow(np.squeeze(true_mask),cmap='gray',alpha=0.3)
      plt.subplot(122)
      plt.title("X-ray image with mask(Predicted)")
      plt.imshow(np.squeeze(image),cmap='gray')
      plt.imshow(np.squeeze(mask),cmap='gray',alpha=0.3)
      return plt.show()
    except: #if there is no ground truth mask
      plt.figure(figsize=(20,6))
      plt.title("X-ray image with mask(Predicted)")
      plt.imshow(np.squeeze(image),cmap='gray')
      plt.imshow(np.squeeze(mask),cmap='gray',alpha=0.3)
      return plt.show()
