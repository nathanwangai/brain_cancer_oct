import os
import numpy as np
import tkinter as tk
import bframe_processing as bfp
import matplotlib.pyplot as plt
import network_processing as ntwp
from PIL import Image, ImageTk
from skimage import img_as_bool
from skimage.transform import resize

class MyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Brain OCT Segmentation")
        self.geometry("700x700")

        self.dir_var = tk.StringVar()
        self.network_var = tk.IntVar()
        self.frame_skip = tk.IntVar()
        self.slice_skip = tk.IntVar()
        self.seg_map = None
        
        self.create_widgets()
        
    def create_widgets(self):

        entry_frame = tk.Frame()
        entry_frame.grid(column=0, row=0, sticky="W")

        # data directory
        entry = tk.Entry(entry_frame, textvariable=self.dir_var)
        entry.insert(tk.END, "Paste Directory")
        entry.grid(column=0, row=0, sticky = "N", padx=(20,0), ipadx=100)

        # sampling rate options
        param1 = tk.Entry(entry_frame, textvariable=self.frame_skip)
        param1.delete(0, tk.END)
        param1.insert(tk.END, "B-frame Sample Rate")
        param1.grid(column=0, row=1, sticky="W", padx=(20,0), ipadx=15)
        param2 = tk.Entry(entry_frame, textvariable=self.slice_skip)
        param2.delete(0, tk.END)
        param2.insert(tk.END, "Slice Sample Rate")
        param2.grid(column=0, row=1, sticky="E", ipadx=15)

        option_frame = tk.Frame()
        option_frame.grid(column=1, row=0, sticky="W")

        # network options
        option1 = tk.Radiobutton(option_frame, text="B-frame CNN", variable=self.network_var, value=1)
        option1.grid(column=0, row=0, sticky="W", padx=(20,0))
        option2 = tk.Radiobutton(option_frame, text="Texture CNN", variable=self.network_var, value=2)
        option2.grid(column=0, row=1, sticky="W", padx=(20,0))
        option3 = tk.Radiobutton(option_frame, text="Ensemble MLP", variable=self.network_var, value=3)
        option3.grid(column=0, row=2, sticky="W", padx=(20,0))
        submit = tk.Button(option_frame, text="Submit", command=self.submit_clicked)
        submit.grid(column=1, row=1, sticky="E", padx=(20,0), ipadx=30)

    def submit_clicked(self):
        # get values from buttons
        dir_path = self.dir_var.get() # D:\\brain_cancer_oct\segmentation_data\\4-12-2021-WM3-1
        frame_skip = int(self.frame_skip.get())
        slice_skip = int(self.slice_skip.get())
        network_choice = self.network_var.get()

        # compute/set relevant parameters
        all_files = os.listdir(dir_path)
        total_frames = len(all_files)
        num_frames = int(np.ceil(total_frames / frame_skip))
        num_slices = int(np.ceil(20 / slice_skip))
        seg_map = np.zeros((num_frames, num_slices))
        counter = 0

        # create image display widget
        self.seg_map = ImageTk.PhotoImage(Image.open("segmap.png").resize((500,500)))
        disp = tk.Label(image=self.seg_map)
        disp.grid(column=0, row=1, columnspan=2, sticky="NSEW")

        # loop over every nth frame
        for i in range(0, total_frames, frame_skip):
            file = all_files[i]
            bframe = bfp.process_bframe(os.path.join(dir_path, file))
            bframe_slices = bfp.slice_bframe(bframe, slice_width=100, step=slice_skip)
            inputs = []
            outputs = None

            # generate inputs according to network choice
            for bframe_slice in bframe_slices:
                edge_depth = bfp.extract_edge(bframe_slice)
                bframe_slice = bframe_slice[edge_depth:edge_depth+200]
                
                if(network_choice == 1): # bframe_cnn
                    inputs.append(bframe_slice)
                elif(network_choice == 2): # texture_cnn
                    texture = bfp.normalize(bfp.convert_to_texture(bframe_slice))
                    inputs.append(texture)
                elif(network_choice == 3): # ensemble_mlp
                    inputs.append(ntwp.concatenate_embeddings(np.squeeze(bframe_slice)))
            inputs = np.array(inputs)

            # generate outputs for slice batch according to network choice
            if(network_choice == 1):
                inputs = np.reshape(inputs, (num_slices,200,100,1))
                outputs = ntwp.get_bframe_predictions(inputs)
            elif(network_choice == 2):
                inputs = np.reshape(inputs, (num_slices,100,100,1))
                outputs = ntwp.get_texture_predictions(inputs)
            elif(network_choice==3):
                inputs = np.reshape(inputs, (num_slices,128))
                outputs = ntwp.get_ensemble_predictions(inputs)
            
            seg_map[counter] = outputs
            counter += 1
            plt.imshow(seg_map, cmap="gray") # replace bframe with segmap
            plt.axis("off")
            plt.savefig("segmap.png", bbox_inches="tight", pad_inches=0)
            plt.close()

            self.seg_map = ImageTk.PhotoImage(Image.open("segmap.png").resize((400,400)))
            disp.configure(image=self.seg_map)
            disp.image = self.seg_map
            self.update()
        
if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
    

    