# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf
model = tf.keras.models.load_model('keras_model.h5')



# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		
		# expand the dimensions
		
		# normalize it before feeding to the model
		
		# get predictions from the model
		img=cv2.resize(frame,(224,224))
    	test_image=np.array(img,dtype=np.float32)
    	test_image=np.expand_dims(test_image,axis=0)
    	normalize_image=test_image/255.0
    	prediction=model.predict(normalize_image)
    	print('Prediction',prediction)
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()

#-----------------------------------------






# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('keras_model.h5')




# define a video capture object
vid = cv2.VideoCapture(0)






while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
  
    img=cv2.resize(frame,(224,224))
    test_image=np.array(img,dtype=np.float32)
    test_image=np.expand_dims(test_image,axis=0)
    normalize_image=test_image/255.0
    prediction=model.predict(normalize_image)
    print('Prediction',prediction)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()