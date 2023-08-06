from imports import *
class LCD_ANN:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1006x500+0+30")
        self.root.resizable(False, False)
        self.root.title("Brain Tumor Detection")
        img4 = Image.open(r"./xray/train/Tumor/Y55.jpg")
        img4 = img4.resize((1006, 500), Image.ANTIALIAS)
        self.photoimg4 = ImageTk.PhotoImage(img4)
        bg_img = Label(self.root, image=self.photoimg4)
        bg_img.place(x=0, y=50, width=1006, height=500)
        title_lbl = Label(text="Brain Tumor Detection", font=(
            "Bradley Hand ITC", 30, "bold"), bg="black", fg="white",)
        title_lbl.place(x=0, y=0, width=1006, height=50)
        self.b1 = Button(text="Import Data", cursor="hand2", command=self.import_data, font=(
            "Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b1.place(x=80, y=130, width=180, height=30)
        self.b3 = Button(text="Train Data", cursor="hand2", command=self.train_data, font=(
            "Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b3.place(x=80, y=180, width=180, height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")
        self.b4 = Button(text="Test Data", cursor="hand2", command=self.test_data, font=(
            "Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b4.place(x=80, y=230, width=180, height=30)
        self.b4["state"] = "disabled"
        self.b4.config(cursor="arrow")

    def import_data(self):
        self.dataDirectory = 'xray/train/'
        self.TumorPatients = os.listdir(self.dataDirectory)
        self.size = 10
        self.NoSlices = 5
        messagebox.showinfo("Import Data", "Data Imported Successfully!")
        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow")
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")

    def train_data(self):
        training_data = []
        for category in categories:
            path = os.path.join(data_dir, category)
            class_num = categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(
                        path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (img_size, img_size))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    pass
        np.random.shuffle(training_data)
        X = []
        y = []
        for features, label in training_data:
            X.append(features.flatten())
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        X = X / 255.0
        model.add(Dense(128, activation='relu', input_dim=X.shape[1]))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        batch_size = 32
        epochs = 10
        model.fit(X, y, batch_size=batch_size,
                  epochs=epochs, validation_split=0.1)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")
        self.b4["state"] = "normal"
        self.b4.config(cursor="hand2")

    def test_data(self):
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        img = ImageTk.PhotoImage(file=filename)
        img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size))
        new_array = new_array.flatten()
        new_array = new_array.reshape(1, -1)
        new_array = new_array / 255.0
        prediction = model.predict(new_array)
        if prediction[0][0] >= 0.5:
            messagebox.showinfo("Test Data","Tumor")
        else:
            messagebox.showinfo("Test Data","Normal")

if __name__ == "__main__":
    root = Tk()
    obj = LCD_ANN(root)
    root.mainloop()
