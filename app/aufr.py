#import tkinter as tk
from tkinter import *
from tkinter import font
import face

def Generate():
    face.generate()
def Train():
    face.train()
def Detect():
    face.detect()

mainWindow = Tk()
mainWindow.minsize(width=400, height=550)
mainWindow.maxsize(width=400, height=500)
mainWindow.title("Face Detector")


font = font.Font(family='Symbol', size=11, weight='normal')

topFrame = Frame(mainWindow, borderwidth=25)
middleFrame = Frame(mainWindow, borderwidth=25)
botttomFrame = Frame(mainWindow, borderwidth=25)
textFrame = Frame(mainWindow, borderwidth=10)

#topFrame.grid(padx=20, pady=20)
textFrame.pack(fill= BOTH)
topFrame.pack(fill= X)
middleFrame.pack(fill= X)
botttomFrame.pack(fill= X)


topButton = Button(topFrame, font=font ,bd=4, text="Load Data", fg="red", width="15", height="3", command = Generate)
middleButton = Button(middleFrame, font=font ,bd=4, text="Train Data", fg="red", width="15", height="3", command = Train)
bottomButton = Button(botttomFrame, font=font ,bd=4, text="Detect Data", fg="red", width="15", height="3", command = Detect)

text = StringVar()
textLabel = Label(textFrame, font=font, textvariable= text, justify=LEFT, padx=5,pady=5, relief =RAISED)
text.set("Hello and Welcome!, \n\n Steps to use. \n 1. Load The images. \n 2. Train the loded Data. \n 3. Now Detect from trained data \n\n @ Md Danish")

textLabel.pack()
topButton.pack()
middleButton.pack()
bottomButton.pack()

mainWindow.mainloop()
