from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tkinter
from tkinter import *
from tkinter import filedialog,Entry
from PIL import Image

def extract_face(filename, required_size=(224,224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat

# GUI part 


global image1
global image2

image1 = None
image2 = None
root = Tk()
root.title("Face Verification App")
head = Label(root, text =  " Verification of two faces", bg = 'green', fg = 'red' , font = ("Bold" , 30)).grid(row = 0 ,column = 0,columnspan = 100,pady= 50 )

lab_image1 = Entry(root, width = 35 ,borderwidth = 5 )
lab_image1.grid(row = 1,column =0 , columnspan = 3, padx = 10 ,pady= 10)
lab_image2 = Entry(root,width = 35 , borderwidth =5 )
lab_image2.grid(row = 1 , column = 5  ,columnspan = 3, padx = 10 ,pady= 10)
def Image1():

    image1 = filedialog.askopenfilename(title = "Select Image 1",filetypes = (("JPG File","*.jpg"),("all files","*.*")))
    lab_image1.delete(0,END)
    lab_image1.insert(0,image1)
def Image2():
    image2 = filedialog.askopenfilename(title = "Select Image 2",filetypes = (("JPG File","*.jpg"),("all files","*.*")))
    lab_image2.delete(0,END)
    lab_image2.insert(0,image2)
    print
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    score = cosine(known_embedding, candidate_embedding)
    if(score <= thresh):
        score = round(score,2)
        lab2 = Label(root,text = "Both the images are same " , font = ("Bold" ,15 ) , bg = "Blue" , fg ="Orange" ).grid(row = 5 , column = 4)
        lab3 = Label(root,text = " with the cosine value of " + str(score) + " ", font = ("Bold" ,15 ) , bg = "Blue" , fg ="Orange" ).grid(row = 6, column = 4)
        lab4 = Label(root,text = " and threshold is 0.5  ", font = ("Bold" ,15 ) , bg = "Blue" , fg ="Orange" ).grid(row = 7 , column = 4)
    else:
        score = round(score,2)
        lab2 = Label(root,text = "Both the images are different " , font = ("Bold" ,15 ) , bg = "Blue" , fg ="Orange" ).grid(row = 5 , column = 4)
        lab3 = Label(root,text = " with the cosine value of " + str(score) + " ", font = ("Bold" ,15 ) , bg = "Blue" , fg ="Orange" ).grid(row = 6, column = 4)
        lab4 = Label(root,text = " and threshold is 0.5  ", font = ("Bold" ,15 ) , bg = "Blue" , fg ="Orange" ).grid(row = 7 , column = 4)
def verify():
    filenames = [lab_image1.get(), lab_image2.get()]
    # get embeddings file filenames
    embeddings = get_embeddings(filenames)
    
    is_match(embeddings[0], embeddings[1])


bt1 = Button(root , text = 'Select Image1 ', command= Image1).grid(row = 2, column = 2)
bt2 = Button(root , text = 'Select Image2 ', command= Image2).grid(row = 2, column = 6)
bt3 = Button(root,text ='Submit',command = verify ).grid(row = 3 ,column = 4)
lab1 = Label(root , text = 'The two faces are' , bg = 'Blue', font = ("Bold", 20)).grid(row = 4 ,column= 4)
bt3 = Button(root,text ='EXIT',command = root.destroy).grid(row = 8 ,column = 4)


root.mainloop()
