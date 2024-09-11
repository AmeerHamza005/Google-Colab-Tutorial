# Google-Colab-Tutorial
### Google Colab Tutorial

Google Colaboratory (Colab) is a free, cloud-based service that allows you to write and execute Python code in Jupyter notebooks without requiring any setup. Colab is a popular tool for data science, machine learning, and deep learning because of its ease of use, cloud-based environment, and access to free GPUs/TPUs.

---

## 1. **What is Google Colab?**

- **Jupyter Notebook Interface:** Colab uses the Jupyter Notebook interface, which makes it easy to combine executable Python code with rich text, images, and visualizations.
- **Cloud-Based:** No installation is required. All work is done online.
- **Free Access to GPUs and TPUs:** Colab provides free access to powerful hardware accelerators like GPUs and TPUs.

---

## 2. **Getting Started with Google Colab**

### **Step 1: Accessing Google Colab**

- Open your web browser and go to [Google Colab](https://colab.research.google.com).
- If you're signed in with a Google account, you can start by opening an existing notebook or creating a new one.

### **Step 2: Creating a New Notebook**

1. Click on **File** > **New notebook**.
2. This will open a blank notebook, where you can start writing and running Python code.

### **Step 3: Basic Notebook Interface**

- **Code Cells:** Use these to write Python code.
  - Example:
    ```python
    print("Hello, Colab!")
    ```
  - Run the code by pressing **Shift + Enter** or by clicking the "Play" button on the left of the cell.
  
- **Text Cells:** These are used for markdown text. You can use these to write notes, instructions, or explanations.  
  - To add a text cell, click on the **+ Text** button in the toolbar.
  - Example Markdown in a text cell:
    ```markdown
    # This is a Heading
    **Bold text** and *italic text*.
    ```
  
- **Saving Work:** Google Colab automatically saves your work to Google Drive. You can manually save by going to **File** > **Save a copy in Drive**.

---

## 3. **Running Python Code in Colab**

### **Basic Python Example**

1. **Hello World:**
   ```python
   print("Hello, World!")
   ```

2. **Importing Libraries:**
   - You can import popular libraries such as NumPy, Pandas, and Matplotlib as you normally would in Python:
     ```python
     import numpy as np
     import pandas as pd
     ```

### **Accessing Google Drive**

You can easily connect Google Colab to your Google Drive to save or load files.

1. Run the following command in a code cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Follow the link provided to authenticate and grant Colab access to your Drive.

### **Installing External Libraries**

Colab allows you to install external libraries using `pip`. For example:
```bash
!pip install seaborn
```

---

## 4. **Working with Datasets**

### **Upload Files**

To upload files directly from your computer:
1. Click on the **Files** icon on the left sidebar.
2. Click the **Upload** button to upload files.

You can also upload files programmatically:
```python
from google.colab import files
uploaded = files.upload()
```

### **Loading Data from Google Drive**

If you have connected Google Drive, you can easily load data from it:
```python
import pandas as pd

# Load a CSV file from Drive
data = pd.read_csv('/content/drive/My Drive/filename.csv')
```

---

## 5. **Using GPUs and TPUs**

Google Colab provides access to free GPUs and TPUs for running machine learning models.

### **Enable GPU or TPU**

1. Go to **Runtime** > **Change runtime type**.
2. Under **Hardware Accelerator**, choose either **GPU** or **TPU**.
3. Click **Save**.

Once the GPU or TPU is enabled, your code can run significantly faster when working with deep learning models.

### **Checking GPU Availability**

To check if Colab is using a GPU:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

---

## 6. **Colab Shortcuts**

Here are some useful keyboard shortcuts:

- **Run Cell:** `Shift + Enter`
- **Insert Code Cell:** `Ctrl + M, B`
- **Insert Text Cell:** `Ctrl + M, A`
- **Delete Cell:** `Ctrl + M, D`
- **Show Command Palette:** `Ctrl + Shift + P`

---

## 7. **Collaborating on Notebooks**

You can share your notebook with others by clicking the **Share** button, just like you would with any Google Docs or Sheets file. Collaborators can view or edit the notebook based on the permissions you provide.

---

## 8. **Visualizing Data in Colab**

Colab supports popular Python data visualization libraries such as Matplotlib, Seaborn, and Plotly.

### **Basic Plot with Matplotlib**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.show()
```

### **Interactive Plots with Plotly**

```python
!pip install plotly
import plotly.express as px

df = px.data.iris()  # Load sample dataset
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
fig.show()
```

---

## 9. **Deep Learning with TensorFlow & Keras in Colab**

Google Colab comes with pre-installed TensorFlow and Keras, which you can use for machine learning projects.

### **Basic Example Using TensorFlow**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
```

---

## 10. **Exporting and Sharing Notebooks**

You can download your notebook as a Python script, HTML file, or Jupyter notebook:

1. Go to **File** > **Download** > and select the desired format:
   - `.ipynb` (Jupyter notebook)
   - `.py` (Python script)
   - `.html` (Web page)

---

## 11. **Advantages of Google Colab**

- **No Setup Required:** You don't need to install Python or any libraries.
- **Access to GPUs/TPUs:** Great for running large deep learning models.
- **Real-Time Collaboration:** Multiple users can work on the same notebook at the same time.
- **Integration with Google Drive:** Save and load data directly from Drive.

---

This tutorial covers the basics of getting started with Google Colab. You can use it to run Python code, train machine learning models, or collaborate on data science projects. Let me know if you need more specific examples or help with anything else!