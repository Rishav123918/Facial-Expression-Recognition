import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('model_file.h5')

faceDetect = cv2.CascadeClassifier('frontalface.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Read the image
frame = cv2.imread("R.jpeg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceDetect.detectMultiScale(gray, 1.3, 3)


# Iterate over detected faces
for x, y, w, h in faces:
    
    # Extract the face region
    sub_face_img = gray[y:y + h, x:x + w]
    
    # Resize the face image to match the input size of the model
    resized = cv2.resize(sub_face_img, (48, 48))
    # Normalize the pixel values
    normalized = resized / 255.0
    # Reshape the image to match the input shape of the model
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    # Make predictions using the model
    result = model.predict(reshaped)

    # Find the predicted label
    label = np.argmax(result, axis=1)[0]
    
     # Get the true label
    # true_label = labels_dict.keys().index(labels_dict[label])
    true_label = labels_dict.get(label)

    # Draw a bounding box around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # Draw a rectangle for the label
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)

    # Put the predicted label on the rectangle
    cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Plot the predicted probabilities for each emotion
    plt.figure(figsize=(10, 6))
    plt.bar(labels_dict.keys(), result.flatten())
    plt.title("Predicted Emotion Probabilities")
    plt.show()
    

# Display the final frame
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# from keras.models import load_model

# model=load_model('model_file.h5')

# faceDetect=cv2.CascadeClassifier('frontalface.xml')

# labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# # len(number_of_image), image_height, image_width, channel

# # frame=cv2.imread("IMG_20231120_214455.jpg")
# frame=cv2.imread("mlti-pic.jpg")
# gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces= faceDetect.detectMultiScale(gray, 1.3, 3)
# for x,y,w,h in faces:
#     sub_face_img=gray[y:y+h, x:x+w]
#     resized=cv2.resize(sub_face_img,(48,48))
#     normalize=resized/255.0
#     reshaped=np.reshape(normalize, (1, 48, 48, 1))
#     result=model.predict(reshaped)
#     label=np.argmax(result, axis=1)[0]
#     print(label)
#     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#     cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#     cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#     cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
# cv2.imshow("Frame",frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



