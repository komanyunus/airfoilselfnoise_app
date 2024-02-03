from tkinter import filedialog,messagebox
from tkinter import *
import ttkbootstrap as tb
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageTk
from tkintertable import TableCanvas, TableModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# !!!! YOU SHOULD WRITE DOWN THE DATA FILE DIRECTORY BELOW



#INSTALLATION OF BASE
root = tb.Window(themename="solar")
#root = Tk()
root.title("Airfoil Self-Noise App");
root.iconbitmap(os.path.join(current_directory, "airblade1.ico"))
root.geometry("400x600")

sgd_var = IntVar()
adam_var = IntVar()

#Data Preparing
#Loading Data
file_path = r"C:\Users\yunus\OneDrive\Masaüstü\Yapay Zeka Yüksek Lisans\ME 524 Artificial Intelligence in Mechanical\Project\airfoil+self+noise\MATLAB\airfoil_self_noise.txt"
# Read the data into a Pandas DataFrame
column_names = ['Frequency', 'AoA', 'ChordLength', 'FreeStreamVelocity', 'SSD_Thickness', 'ScaledSoundPressureLevel']
df_data = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
#Normalization
numerical_columns = ['Frequency', 'AoA', 'ChordLength','SSD_Thickness', 'ScaledSoundPressureLevel']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply normalization to the numerical columns
df_data[:] = scaler.fit_transform(df_data[:])
# 2. Split Data into Features and Target
X = df_data[['Frequency', 'AoA', 'ChordLength','FreeStreamVelocity', 'SSD_Thickness']]
y = df_data['ScaledSoundPressureLevel']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

#NASA IMAGE
img_lbl1_x, img_lbl1_y = 75,35
img_nasa1 = Image.open(os.path.join(current_directory, "NASA_logo.svg.png"))
img_nasa1 = img_nasa1.resize((245, 205), Image.LANCZOS)  # Resize the image
img_nasa1 = ImageTk.PhotoImage(img_nasa1)
img_lbl1 = tb.Label(root,image=img_nasa1)
img_lbl1.place(x = img_lbl1_x, y = img_lbl1_y )

img_lbl2 = tb.Label(root,text="Airfoil Self-Noise",font=("Times",28,"bold italic underline"))
img_lbl2.place(x = img_lbl1_x , y = img_lbl1_y + 200)

#Create NN Model
model= Sequential()
model.add(Dense(512, input_dim=X_train.shape[1]))
poly_model = LinearRegression()


def Add_Layer(node,acfunc):
    global model,network_tree
    activation_list = [None,"sigmoid","relu","tanh","softmax"]
    x = Dense(int(node.get()),activation=activation_list[acfunc.current()])
    model.add(x)
    # Build the model before calling summary
    model.build(input_shape=(1,5))  
    model.summary()
    
    #Adding to treeview
    network_tree.insert("",END,values= (int(node.get()),acfunc.get()))

def Remove_Layer():
    global model,network_tree
    model.pop()
    # Build the model before calling summary
    model.build(input_shape=(1,5))  
    model.summary()
    children = network_tree.get_children()
    if children:  # Check if there are any items in the treeview
        last_item = children[-1]  # Get the identifier of the last item
        network_tree.delete(last_item)  # Delete the last item
    else:
        print("The treeview is empty, cannot delete any item.")


def switchmakesure1(switch1,switch2):
    global sgd_var,adam_var
    if sgd_var.get() == 1:
        adam_var.set(0)
    elif adam_var.get() == 1:
        sgd_var.set(0)
        
    switch1.config()
    switch2.config()

def switchmakesure2(switch1,switch2):
    global sgd_var,adam_var
    if adam_var.get() == 1:
        sgd_var.set(0)
    elif sgd_var.get() == 1:
        adam_var.set(0)
    switch1.config()
    switch2.config()
    
def start_train():
    global sgd_var,adam_var,lr_entry,epc_entry,model,history,batch_entry,val_cycle,X_train,y_train,plot_frame
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    
    lr= float(lr_entry.get())
    epc = int(epc_entry.get())
    batch = int(batch_entry.get())
    val_split = float(val_cycle.amountusedvar.get())/100.0
    
    if sgd_var.get() == 1 and adam_var.get() == 0:
        opt = keras.optimizers.SGD(learning_rate=lr)
        
    elif adam_var.get() == 1 and sgd_var.get() == 0:
        opt = keras.optimizers.Adam(learning_rate=lr)
    
    else:
        print("Select an optimizer")
        
    model.compile(optimizer=opt, loss='mean_squared_error',  metrics=['mae',keras.metrics.RootMeanSquaredError()])
    history = model.fit(
    X_train, y_train,
    epochs=epc, batch_size=batch,
    validation_split=val_split,  # 20% of training data will be used as a validation set
    callbacks=keras.callbacks.EarlyStopping(patience=100)
    )
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    # Create a FigureCanvasTkAgg object to embed the plot in the Tkinter frame
    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()

    # Attach the canvas to the Tkinter frame
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    
def poly_train():
    global deg_cycle,X_train,y_train,poly_model,plot_frame2,poly
    
    for widget in plot_frame2.winfo_children():
        widget.destroy()
    
    degree = int(deg_cycle.amountusedvar.get())
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    poly_model.fit(X_train_poly, y_train)
    y_train_pred = poly_model.predict(X_train_poly)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Create a table with error metrics
    table_data = [
        ["RMSE", "MAE", "R^2"],
        [train_rmse, train_mae, train_r2]
    ]

    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off the axis for a cleaner look
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLabels=None, rowLabels=None)

    # Embed the table into the frame
    canvas = FigureCanvasTkAgg(fig, master=plot_frame2)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    
def plot_actpredorbar(isbar):
    global plot_frame3, model, poly, X_test, y_test,y_pred_nn,y_pred_poly
    for widget in plot_frame3.winfo_children():
        widget.destroy()

    # Neural Network Model Predictions
    y_pred_nn = model.predict(X_test).flatten()

    # Polynomial Regression Model Predictions
    X_test_poly = poly.transform(X_test)
    y_pred_poly = poly_model.predict(X_test_poly)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    if not isbar:
        # Find the common y-axis limits for proper comparison
        y_min = min(y_test.min(), y_pred_nn.min(), y_pred_poly.min())
        y_max = max(y_test.max(), y_pred_nn.max(), y_pred_poly.max())

        # Neural Network Model
        ax.scatter(y_test, y_pred_nn, color='blue')
        ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=2)
        ax.set_xlabel('Actual Scaled Sound Pressure Level')
        ax.set_ylabel('Predicted Scaled Sound Pressure Level')
        ax.set_title('Neural Network Model - Prediction vs Actual')
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)

        # Polynomial Regression Model
        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(y_test, y_pred_poly, color='green')
        ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=2)
        ax.set_xlabel('Actual Scaled Sound Pressure Level')
        ax.set_ylabel('Predicted Scaled Sound Pressure Level')
        ax.set_title('Polynomial Regression Model - Prediction vs Actual')
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)
    else:
        # Calculate evaluation metrics for Neural Network Model
        mse_nn = mean_squared_error(y_test, y_pred_nn)
        mae_nn = mean_absolute_error(y_test, y_pred_nn)
        rmse_nn = np.sqrt(mse_nn)

        # Calculate evaluation metrics for Polynomial Regression Model
        mse_poly = mean_squared_error(y_test, y_pred_poly)
        mae_poly = mean_absolute_error(y_test, y_pred_poly)
        rmse_poly = np.sqrt(mse_poly)

        # Create bar chart
        metrics = ['MSE', 'MAE', 'RMSE']
        nn_values = [mse_nn, mae_nn, rmse_nn]
        poly_values = [mse_poly, mae_poly, rmse_poly]

        bar_width = 0.35
        index = np.arange(len(metrics))

        ax.bar(index, nn_values, bar_width, label='Neural Network', color='blue')
        ax.bar(index + bar_width, poly_values, bar_width, label='Polynomial Regression', color='green')

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Comparison of Evaluation Metrics between Models')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(metrics)
        ax.legend()

    # Ensure proper layout of subplots
    plt.tight_layout()

    # Embed the figure into the frame
    canvas = FigureCanvasTkAgg(fig, master=plot_frame3)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        
    
        
def NN_Window():
    global network_tree,sgd_var,adam_var,lr_entry,epc_entry,batch_entry,val_cycle,plot_frame
    
    nn_win = tb.Toplevel()
    nn_win.title("Neural Network Training")
    nn_win.iconbitmap(os.path.join(current_directory, "airblade1.ico"))
    nn_win.geometry("1024x768")
    
    #Adding Dense Layer
    nodes_lbl_x, nodes_lbl_y = 50, 50
    nodes_lbl = tb.Label(nn_win,text="NODES",font=("Times",20))
    nodes_lbl.place(x=nodes_lbl_x,y=nodes_lbl_y)
    
    nodes_entry = tb.Entry(nn_win)
    nodes_entry.place(x = nodes_lbl_x, y= nodes_lbl_y+50)
    
    #Activation Function
    func_lbl_x, func_lbl_y = 50, 150
    func_lbl = tb.Label(nn_win,text="FUNCTION",font=("Times",20))
    func_lbl.place(x=func_lbl_x,y=func_lbl_y)
    
    activation_list = ["No Function","Sigmoid","ReLU","TanH","Softmax (for Clasifier)"]
    
    func_combo = tb.Combobox(nn_win,values=activation_list)
    func_combo.place(x = func_lbl_x, y= func_lbl_y + 50)
    func_combo.current(0)

    #Network Visualizer
    network_lbl_x, network_lbl_y = func_lbl_x, func_lbl_y + 100
    network_lbl = tb.Label(nn_win,text="NETWORK",font=("Times",20))
    network_lbl.place(x=network_lbl_x,y=network_lbl_y)
    
    network_col = ("Nodes","Activation Function")
    network_tree = tb.Treeview(nn_win,columns=network_col,show="headings")
    network_tree.heading("Nodes", text= "Nodes")
    network_tree.heading("Activation Function",text= "Activation Function")
    network_tree.column("Nodes", width=70)
    network_tree.column("Activation Function", width=119)
    network_tree.place(x=network_lbl_x-25,y=network_lbl_y + 50,width=200)
    
    #Add Layer Button
    addlayer_btn = tb.Button(nn_win,text="Add Layer",command=lambda :Add_Layer(nodes_entry,func_combo),bootstyle="success");
    addlayer_btn.place(x = network_lbl_x + 25, y = network_lbl_y+250)
    
    #Remove Layer Button
    removelayer_btn = tb.Button(nn_win,text="Remove Layer",command=Remove_Layer,bootstyle="danger");
    removelayer_btn.place(x = network_lbl_x + 25, y = network_lbl_y+300)
    
    #Learning Rate
    lr_entry_x, lr_entry_y = nodes_lbl_x + 300, nodes_lbl_y
    lr_label = tb.Label(nn_win,text="Learning Rate",font=("Times",16))
    lr_label.place(x = lr_entry_x, y= lr_entry_y)
    lr_entry = tb.Entry(nn_win)
    lr_entry.place(x = lr_entry_x, y= lr_entry_y + 50)
    
    #Epoch
    epc_entry_x, epc_entry_y = nodes_lbl_x + 450, nodes_lbl_y
    epc_label = tb.Label(nn_win,text="Epoch",font=("Times",16))
    epc_label.place(x = epc_entry_x, y= epc_entry_y)
    epc_entry = tb.Entry(nn_win)
    epc_entry.place(x = epc_entry_x, y= epc_entry_y + 50)
    
    #Batch Size
    batch_entry_x, batch_entry_y = nodes_lbl_x + 600, nodes_lbl_y
    batch_label = tb.Label(nn_win,text="Batch Size",font=("Times",16))
    batch_label.place(x = batch_entry_x, y= batch_entry_y)
    batch_entry = tb.Entry(nn_win)
    batch_entry.place(x = batch_entry_x, y= batch_entry_y + 50)
    
    #Validation Split Cycle
    val_cycle_x ,val_cycle_y = nodes_lbl_x + 750, nodes_lbl_y

    val_cycle = tb.Meter(nn_win, bootstyle="danger",
                           subtext="Val_Rate",
                           subtextfont="-size 8",
                           meterthickness=3,
                           textfont="-size 12",
                           interactive=True,
                           metertype="semi",
                           metersize=120,
                           amounttotal=45
                           )
    val_cycle.place(x=val_cycle_x  ,y = val_cycle_y)
    
    #Optimizer Selection

    
    sgd_button_x ,sgd_button_y = nodes_lbl_x + 150, nodes_lbl_y
    sgd_button = tb.Checkbutton(nn_win,bootstyle="success-round-toggle", text="SGD Optimizer",variable=sgd_var,onvalue=1,offvalue=0,command=lambda:switchmakesure1(sgd_button,adam_button))
    sgd_button.place(x = sgd_button_x ,y = sgd_button_y)
    
    adam_button = tb.Checkbutton(nn_win,bootstyle="success-round-toggle", text="Adam Optimizer",variable=adam_var,onvalue=1,offvalue=0,command=lambda:switchmakesure2(adam_button,sgd_button))
    adam_button.place(x = sgd_button_x ,y = sgd_button_y + 25)
    
    # Create a new frame for plotting
    plot_frame_x, plot_frame_y = 300,300
    plot_frame = Frame(nn_win)
    plot_frame.place(x=plot_frame_x, y=plot_frame_y, width=600, height=400)
     #Start Training
     #Train Button
    train_button_x ,train_button_y = nodes_lbl_x , nodes_lbl_y + 600
    train_button = tb.Button(nn_win, text="Start Training",command=start_train,bootstyle="success")
    train_button.place(x = train_button_x ,y = train_button_y)
    

     
def poly_Window():
    global deg_cycle,plot_frame2
    poly_win = tb.Toplevel()
    poly_win.title("Polynomial Regression")
    poly_win.iconbitmap(os.path.join(current_directory, "airblade1.ico"))
    poly_win.geometry("800x600")
    
    #Polynomial Degree Cycle
    deg_cycle_x ,deg_cycle_y = 50,50

    deg_cycle = tb.Meter(poly_win, bootstyle="danger",
                           subtext="Degree",
                           subtextfont="-size 8",
                           meterthickness=3,
                           textfont="-size 12",
                           interactive=True,
                           metertype="semi",
                           metersize=120,
                           amounttotal=10
                           )
    deg_cycle.place(x=deg_cycle_x  ,y = deg_cycle_y)
    
    # Create a new frame for plotting
    plot_frame2_x, plot_frame2_y = 200,100
    plot_frame2 = Frame(poly_win)
    plot_frame2.place(x=plot_frame2_x, y=plot_frame2_y, width=600, height=400)
    
    polyreg_btn = tb.Button(poly_win,text="Get Train Errors",command=poly_train,bootstyle="success");

    polyreg_btn_x, polyreg_btn_y =  50,200
    polyreg_btn.place(x=polyreg_btn_x,y=polyreg_btn_y)
    
    
def eval_Window():
    global plot_frame3
    eval_win = tb.Toplevel()
    eval_win.title("Evaluation Results")
    eval_win.iconbitmap(os.path.join(current_directory, "airblade1.ico"))
    eval_win.geometry("800x600")
    
    # Create a new frame for plotting
    plot_frame3_x, plot_frame3_y = 50,100
    plot_frame3 = Frame(eval_win)
    plot_frame3.place(x=plot_frame3_x, y=plot_frame3_y, width=700, height=400)
    
    actpred_button_x ,actpred_button_y = 50,50
    actpred_button = tb.Button(eval_win, text="Actual vs Prediction",command=lambda:plot_actpredorbar(False),bootstyle="success")
    actpred_button.place(x = actpred_button_x ,y = actpred_button_y)
    
    bar_button_x ,bar_button_y = 200,50
    bar_button = tb.Button(eval_win, text="Errors Bar Chart",command=lambda:plot_actpredorbar(True),bootstyle="success")
    bar_button.place(x = bar_button_x ,y = bar_button_y)

     
     
    
NN_btn = tb.Button(root,text="NEURAL NETWORK TRAINING",command=NN_Window,bootstyle="success");

NN_btn_x, NN_btn_y = int((400-NN_btn.winfo_reqwidth())/2), img_lbl1_y + 300  
NN_btn.place(x=NN_btn_x,y=NN_btn_y)


poly_btn = tb.Button(root,text="POLYNOMIAL REGRESSION",command=poly_Window,bootstyle="success");

poly_btn_x, poly_btn_y = int((400-poly_btn.winfo_reqwidth())/2), img_lbl1_y + 350  
poly_btn.place(x=poly_btn_x,y=poly_btn_y)

ev_btn = tb.Button(root,text="EVALUATION RESULTS",command=eval_Window,bootstyle="success");

ev_btn_x, ev_btn_y = int((400-ev_btn.winfo_reqwidth())/2), img_lbl1_y + 400  
ev_btn.place(x=ev_btn_x,y=ev_btn_y)


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        sys.exit()
# Bind the closing event to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()