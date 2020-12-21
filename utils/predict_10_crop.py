'''
crop file
'''
import numpy as np
import tensorflow as tf

'''
example for using:
x1 = np.zeros(shape=(len(x),1,12,12,3))
for i in range(len(x)):
    x1[i] = crop_image(x[i],crop_type='random', crop_height = 12, crop_width = 12)
'''

def crop_image(image, crop_type,crop_height,crop_width):
    
    '''
    inputs:
    parameter image: has shape (image_height, image_width, num_filters)
    parameter crop_type: whether random(randomly choosing a part of image) 
                                 or central(choose the central part of iamge)
                                 or tencrop(choose central,leftup,rightup,leftdown,rightdown of image
                                            and these five part of the horizontal flipped image)
    parameter crop_height: height of the cropped image
    parameter crop_width: width of the cropped image
    
    outputs:
    img: numpy array with shape(1,crop_height,crop_width,num_filters) if crop_type = random or central
                          shape(10,crop_height,crop_width,num_filters) if crop_type = tencrop
    '''
    
    h_org = image.shape[0]
    w_org = image.shape[1]
    filters = image.shape[2]
    
    
    if crop_type=='random':
        
        img = np.zeros(shape=(1,crop_height,crop_width,filters))

        index_h = np.random.randint(0, high=h_org-crop_height, size=1)[0]
        index_w = np.random.randint(0, high=w_org-crop_width, size=1)[0]
        
        for i in range(filters):
            img[:,:,:,i] = image[index_h:(index_h+crop_height),index_w:(index_w+crop_width),i]
    
    elif crop_type=='central':
        
        img = np.zeros(shape=(1,crop_height,crop_width,filters))

        index_h = int((h_org-crop_height)/2)
        index_w = int((w_org-crop_width)/2)
        
        for i in range(filters):
            img[:,:,:,i] = image[index_h:(index_h+crop_height),index_w:(index_w+crop_width),i]
    
    elif crop_type=='tencrop':
        index_h = int((h_org-crop_height)/2)
        index_w = int((w_org-crop_width)/2)
        
        img = np.zeros(shape=(10,crop_height,crop_width,filters))

        image_flip = np.fliplr(image)
        
        for i in range(filters):
            img[0,:,:,i] = image[index_h:(index_h+crop_height),index_w:(index_w+crop_width),i]
            img[1,:,:,i] = image[:crop_height,:crop_width,i]
            img[2,:,:,i] = image[:crop_height,-crop_width:,i]
            img[3,:,:,i] = image[-crop_height:,:crop_width,i]
            img[4,:,:,i] = image[-crop_height:,-crop_width:,i]
            
            img[5,:,:,i] = image_flip[index_h:(index_h+crop_height),index_w:(index_w+crop_width),i]
            img[6,:,:,i] = image_flip[:crop_height,:crop_width,i]
            img[7,:,:,i] = image_flip[:crop_height,-crop_width:,i]
            img[8,:,:,i] = image_flip[-crop_height:,:crop_width,i]
            img[9,:,:,i] = image_flip[-crop_height:,-crop_width:,i]
    
    return img

# 10 crop testing
def predict_10_crop(img_gen, model, n):
    all_preds = []
    all_top_n_preds = []
    i = 0
    for img, _ in img_gen:
        if i == n:
            return np.array(all_preds), np.array(all_top_n_preds)
        else:
            crop_img = crop_image(img.reshape([64,64,3]), crop_type='tencrop', crop_height = 56, crop_width = 56)
            crop_img = tf.image.resize(crop_img, [64,64])
            y_pred = model.predict(crop_img).mean(axis=0)
            preds = np.argmax(y_pred)
            top_n_preds= np.argpartition(y_pred, -5)[-5:]

            all_preds.append(preds)
            all_top_n_preds.append(top_n_preds)
            i+=1
        
    
