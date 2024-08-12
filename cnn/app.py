import pygame, sys
from pygame.locals import *
import numpy as np
import keras 
from keras.models import load_model
import cv2

# definitions that are needed for the project
sizeX = 640
sizeY = 480

Boundry = 5


Cwhite=(255,255,255)
Cblack= (0,0,0)
Cred=(255,0,0)
predict=True
ImageSave= False
image_cnt= 0

Model= load_model("C://Project//AI//mnist//cnn//bestmodel.h5") 
labels= {0:"Zero",
         1:"One",
         2:"Two",
         3:"Three",
         4:"Four",
         5:"Five",
         6:"Six",
         7:"Seven",
         8:"Eight",
         9:"Nine"
         }
# Initialize pygame 
pygame.init()

Font = pygame.font.Font('freesansbold.ttf',18)
DisplaySurf= pygame.display.set_mode((sizeX,sizeY))
pygame.display.set_mode((sizeX,sizeY))

pygame.display.set_caption("Digit Board")

isWriting = False
Number_xcord=[]
Number_ycord=[]


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and isWriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DisplaySurf,Cwhite,(xcord,ycord), 4,0)
            Number_xcord.append(xcord)
            Number_ycord.append(ycord)
        
        if event.type == MOUSEBUTTONDOWN:
            isWriting = True
            
        if event.type == MOUSEBUTTONUP:
            isWriting = False
            Number_xcord=sorted(Number_xcord)
            Number_ycord=sorted(Number_ycord)
            
            rect_min_x,rect_max_x = max(Number_xcord[0]-Boundry,0),min(sizeX,Number_xcord[-1]-Boundry)
            rect_min_y,rect_max_y = max(Number_ycord[0]-Boundry,0),min(sizeY,Number_ycord[-1]-Boundry)
            
            Number_xcord=[]
            Number_ycord=[]
            
            img_arr = np.array(pygame.PixelArray(DisplaySurf))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            
            if ImageSave:
                cv2.imwrite("image.png")
                image_cnt+=1
            
            if predict:
                
                image= cv2.resize(img_arr,(28,28))
                image= np.pad(image,(10,10),'constant',constant_values=0)
                image= cv2.resize(image,(28,28))/255
                
                label=str(labels[np.argmax(Model.predict(image.reshape(1,28,28,1)))])
                textsurface= Font.render(label,True,Cred,Cwhite)
                textRecObj = textsurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                DisplaySurf.blit(textsurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DisplaySurf.fill(Cblack)

                
    pygame.display.update()