import tkinter as tk # Import tkinter, python GUI application
from tkinter import *
from tkinter import ttk
import os

# All GUI parameters

# GUI main window
main = tk.Tk() 
main.title('LRI Control Simulation Interface')
main_width = 1200
main_length = 700
main_size = (str)(main_width) + "x" + (str)(main_length)
main.geometry(main_size)

# Rocket Selection Widget

# Rocket Label
rocket = Label(main, text = "Rocket:")
rocket.config(font =("Georgia", 14))
rocket.place(x = 5, y = 5)

# Retreive all current rockets
current_dir = os.getcwd()

while (current_dir[-3:] != 'LRI'): # Checks last three letters to ensure you are in your LRI directory
    os.chdir('..')
    current_dir = os.getcwd()


relative_path = os.path.join(current_dir, "FV-Controls/rockets")
os.chdir(relative_path)
current_dir = os.getcwd()

rockets = []
with os.scandir(current_dir) as entries:
      for entry in entries:
          if entry.is_dir():
             rockets.append(entry.name)
rockets.append("New Rocket")

# Allow user to create a new rocket
def newRocket(event):
    if (rocketSelect.get() == "New Rocket"):
        rocketSelect.place_forget()
        name_entry.place(x = 55, y = 5)

def returnOption(event):
    # Add rocket and create new directories
    new_rocket = name_entry.get()
    rockets.append(new_rocket)
    os.mkdir(new_rocket)

    relative_path = os.path.join(current_dir, new_rocket)
    os.chdir(relative_path)
    os.mkdir('data')
    os.mkdir('rocketpy')
    os.mkdir('sensors')
    os.mkdir('simulations')
    os.chdir('..')
    name_entry.place_forget()
    rocketSelect.configure(values=rockets)
    rocketSelect.set()
    rocketSelect.place(x = 55, y = 5)

# Combobox  
rocketSelect = ttk.Combobox(main, values=rockets, width = 10)
rocketSelect.set("---")
rocketSelect.place(x = 55, y = 5)
rocketSelect.bind('<<ComboboxSelected>>', newRocket)

# Text Entry for New Rocket Name
name_var = tk.StringVar()
name_entry = tk.Entry(main,textvariable = name_var, font=('Georgia',14,'normal'))
name_entry.bind("<Return>", returnOption)

# Opens Window
main.mainloop() 