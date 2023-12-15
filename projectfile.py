

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score

import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle


# In[ ]:


x_train=[]
y_train=[]
image_size=150
labels=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
  folderpath = os.path.join(r'C:\Users\Satyajit\OneDrive\Desktop\final year project\cerebral neoplasm dataset\Training', i)

  for j in os.listdir(folderpath):
    img = cv2.imread(os.path.join(folderpath,j))
    img = cv2.resize(img, (image_size, image_size))
    x_train.append(img)
    y_train.append(i)

for i in labels:
  folderpath = os.path.join('C:\\Users\\Satyajit\\OneDrive\\Desktop\\final year project\\cerebral neoplasm dataset\\Training', i)

  for j in os.listdir(folderpath):
    img = cv2.imread(os.path.join(folderpath,j))
    img = cv2.resize(img, (image_size, image_size))
    x_train.append(img)
    y_train.append(i)


# In[ ]:


x_train=np.array(x_train)
y_train=np.array(y_train)


# In[ ]:


## **Shuffling**


# In[ ]:


x_train,y_train=shuffle(x_train,y_train,random_state=101)
x_train.shape


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.1,random_state=101)

y_train_new=[]
for i in y_train:
  y_train_new.append(labels.index(i))
y_train=y_train_new
y_train=tf.keras.utils.to_categorical(y_train)


y_test_new=[]
for i in y_test:
  y_test_new.append(labels.index(i))
y_test=y_test_new
y_test=tf.keras.utils.to_categorical(y_test)


# In[ ]:


model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[ ]:


history =model.fit(x_train,y_train,epochs=1,validation_split=0.1)


# In[ ]:



import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


model.save('my_model.keras')
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs=range(len(acc))
fig=plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()


# In[ ]:


model.save('my_model.keras')
# Save the model to a file
model.save('your_model_name.h5')

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(loss))
fig=plt.figure(figsize=(14,7))
plt.plot(epochs,loss,'r',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.legend(loc='upper left')
plt.show()


# In[ ]:


img = cv2.imread(r'C:\Users\Satyajit\OneDrive\Desktop\final year project\cerebral neoplasm dataset\Training\pituitary_tumor\p (1).jpg')

if img is not None:
    # Resize the image
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array.shape
else:
    print("Failed to load the image.")


# In[ ]:


img_array=img_array.reshape(1,150,150,3)
img_array.shape


# In[ ]:


from tensorflow.keras.preprocessing import image


img = image.load_img(r'C:\Users\Satyajit\OneDrive\Desktop\final year project\cerebral neoplasm dataset\Training\no_tumor\1.jpg')

plt.imshow(img,interpolation="nearest")
plt.show()


# In[ ]:


a=model.predict(img_array)
indices=a.argmax()
indices


# In[ ]:


label=labels[indices]
print("Predicted label:",label)



# Flask app to allow downloading the model
app = Flask(__name__)

@app.route('/download_model')
def download_model():
    return send_file('my_model.h5', as_attachment=True)

if __name__ == '__main__':
    app.run()
# # In[ ]:


# from flask import Flask, request, render_template_string
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# import io

# app = Flask(__name__)
# model = load_model('my_model.keras')
# labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         # Form submitted, process the image and show result on the same page
#         if 'file' not in request.files:
#             return "No file part"

#         file = request.files['file']
#         if file.filename == '':
#             return "No selected file"
#         if file:
#             img = Image.open(io.BytesIO(file.read()))
#             img = img.resize((150, 150))
#             img = image.img_to_array(img)
#             img = img.reshape(1, 150, 150, 3)
#             result = model.predict(img)
#             predicted_class = labels[np.argmax(result)]
#             return render_template_string("""
#                 <!DOCTYPE html>
#                 <html lang="en">
#                 <head>
#                     <meta charset="UTF-8">
#                     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#                     <title>Cerebral Neoplasm Detection</title>
#                     <style>
#                         body {
#                             font-family: Arial, sans-serif;
#                             background-color: #f4f4f4;
#                             margin: 0;
#                             padding: 0;
#                             display: flex;
#                             flex-direction: column;
#                             align-items: center;
#                             height: 100vh;
#                         }

#                         h1 {
#                             text-align: center;
#                             color: #333;
#                             margin-bottom: 20px;
#                         }

#                         form {
#                             text-align: center;
#                         }

#                         input[type="file"] {
#                             padding: 10px;
#                             margin-bottom: 10px;
#                             border: 1px solid #ddd;
#                             border-radius: 5px;
#                             background-color: #fff;
#                             cursor: pointer;
#                         }

#                         input[type="submit"] {
#                             padding: 10px 20px;
#                             background-color: #4caf50;
#                             color: #fff;
#                             border: none;
#                             border-radius: 5px;
#                             cursor: pointer;
#                         }

#                         input[type="submit"]:hover {
#                             background-color: #45a049;
#                         }

#                         p {
#                             margin-top: 20px;
#                             font-size: 18px;
#                             color: #333;
#                         }
#                     </style>
#                 </head>
#                 <body>
#                     <h1>Cerebral Neoplasm Detection</h1>
#                     <form action="/" method="post" enctype="multipart/form-data">
#                         <input type="file" name="file" accept=".jpg, .jpeg, .png">
#                         <br>
#                         <input type="submit" value="Predict">
#                     </form>
#                     <p>Prediction: {{ predicted_class }}</p>
#                 </body>
#                 </html>
#             """, predicted_class=predicted_class)

#     # Render the initial page
#     return render_template_string("""
#         <!DOCTYPE html>
#         <html lang="en">
#         <head>
#             <meta charset="UTF-8">
#             <meta name="viewport" content="width=device-width, initial-scale=1.0">
#             <title>Cerebral Neoplasm Detection</title>
#             <style>
#                 body {
#                     font-family: Arial, sans-serif;
#                     background-color: #f4f4f4;
#                     margin: 0;
#                     padding: 0;
#                     display: flex;
#                     flex-direction: column;
#                     align-items: center;
#                     height: 100vh;
#                 }

#                 h1 {
#                     text-align: center;
#                     color: #333;
#                     margin-bottom: 20px;
#                 }

#                 form {
#                     text-align: center;
#                 }

#                 input[type="file"] {
#                     padding: 10px;
#                     margin-bottom: 10px;
#                     border: 1px solid #ddd;
#                     border-radius: 5px;
#                     background-color: #fff;
#                     cursor: pointer;
#                 }

#                 input[type="submit"] {
#                     padding: 10px 20px;
#                     background-color: #4caf50;
#                     color: #fff;
#                     border: none;
#                     border-radius: 5px;
#                     cursor: pointer;
#                 }

#                 input[type="submit"]:hover {
#                     background-color: #45a049;
#                 }
#             </style>
#         </head>
#         <body>
#             <h1>Cerebral Neoplasm Detection</h1>
#             <form action="/" method="post" enctype="multipart/form-data">
#                 <input type="file" name="file" accept=".jpg, .jpeg, .png">
#                 <br>
#                 <input type="submit" value="Predict">
#             </form>
#         </body>
#         </html>
#     """)

# if __name__ == '__main__':
#     app.run()


# # In[ ]:




