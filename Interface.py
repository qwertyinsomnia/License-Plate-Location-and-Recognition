import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
from tkinter import StringVar
import cv2
from PIL import Image, ImageTk
import threading
import time

from PlateRecognition import MODEL_CNN, MODEL_SVM
import PlateRecognition
import Train

class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    screen_pos_x = 200
    screen_pos_y = 100
    update_time = 0
    thread = None
    thread_run = False
		
    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)

        win.title("PlateRecognition")
        screen_resolution = '1280x700' + '+' + str(self.screen_pos_x) + '+' + str(self.screen_pos_y)
        win.geometry(screen_resolution)

        # Load model
        self.cnnTrainer = Train.CNNTrainer()
        self.cnnTrainer.loadKNNDataAndTrainKNN()
        self.cnnModel = self.cnnTrainer.kNearest
        self.svmTrainer = Train.Trainer()
        self.svmModel = self.svmTrainer.train_svm()
        self.model = self.cnnModel # default choose cnn
        self.modelType = MODEL_CNN
        
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side="left",expand=1,fill="both")
        frame_right1.pack(side="top",expand=1,fill=tk.Y)
        frame_right2.pack(side="right",expand=0)
        ttk.Label(frame_left, text='src image:').pack(anchor="nw") 
        ttk.Label(frame_right1, text='license plate position:').grid(column=0, row=0, sticky=tk.W)
        ttk.Label(frame_right2, text='choose model:').pack(anchor="se") 
        
        data = ["kNearest", "SVM"]
        self.value = StringVar()
        self.value.set("kNearest")
        self.model_ctl = ttk.Combobox(frame_right2, width=17, height=8, state='normal', textvariable=self.value)
        self.model_ctl.bind("<<ComboboxSelected>>", self.SetModel)
        self.model_ctl["values"] = data
        self.model_ctl.current(0)
        self.model_ctl.pack()

        from_pic_ctl = ttk.Button(frame_right2, text="load image", width=20, command=self.from_pic)
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        start_recog_ctl = ttk.Button(frame_right2, text="start", width=20, command=self.start_recognition)
        
        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='result:').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        # self.color_ctl = ttk.Label(frame_right1, text="12321", width="20")
        # self.color_ctl.grid(column=0, row=4, sticky=tk.W)

        from_pic_ctl.pack(anchor="se", pady="5")
        start_recog_ctl.pack(anchor="se", pady="5")
        

    def start_recognition(self):
        r, roi = PlateRecognition.PlateRecognition(self.pic_path, False, self.model, self.modelType)
        self.show_roi(r, roi)

    def SetModel(self, event):
        model = self.value.get()
        # print('选中的数据:{}'.format(self.model_ctl.get()))
        # print('value的值:{}'.format(self.value.get()))
        # print(model)
        if model == "kNearest":
            self.model = self.cnnModel
            self.modelType = MODEL_CNN
            print("select model: kNearest")
        elif model == "SVM":
            self.model = self.svmModel
            self.modelType = MODEL_SVM
            print("select model: SVM")

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            
            wide = int(wide * factor)
            if wide <= 0 : wide = 1
            high = int(high * factor)
            if high <= 0 : high = 1
            im=im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk
    
    def show_roi(self, r, roi):
        if r :
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))
            self.update_time = time.time()
        
    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="choose the image", filetypes=[('All files','*.*'), ("jpg", "*.jpg"), ('png','*.png')])
        if self.pic_path:
            # r, roi = PlateRecognition.PlateRecognition(self.pic_path, False, self.model, self.modelType)
            img_bgr = cv2.imread(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            # self.show_roi(r, roi)

def close_window():
    print("destroy")
    if surface.thread_run :
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win=tk.Tk()
    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()