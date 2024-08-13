import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.utils import set_random_seed
from keras.backend import clear_session
from PIL import Image
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
X_train = X_train / 255
X_test = X_test / 255
from keras.utils import to_categorical
st.title('MNIST-Fashion Classification')

tab1,tab2 = st.tabs(['Train', 'Inference'])
with tab1:
        col1,col2 = st.columns(2)
        with col1:
            fig, axs = plt.subplots(10, 10)
            fig.set_figheight(8)
            fig.set_figwidth(8)
            for i in range(10):
                ids = np.where(y_train == i)[0]
                for j in range(10):
                    target = np.random.choice(ids)
                    axs[i][j].axis('off')
                    axs[i][j].imshow(X_train[target], cmap='gray')
            st.text('Dataset')
            st.pyplot(fig)
        with col2:
            y_train_ohe = to_categorical(y_train, num_classes=10)
            y_test_ohe = to_categorical(y_test, num_classes=10)
            clear_session()
            set_random_seed(42)

            model = Sequential()
            model.add(Input(shape=X_train.shape[1:]))
            model.add(Flatten())
            model.add(Dense(10, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            Epochs = st.number_input('Echops', value=10, min_value=1,step=1)
            if st.button('Train',use_container_width=True):
                with st.spinner('Training'):
                    history = model.fit(X_train, y_train_ohe, epochs = Epochs, verbose=0)
                loss, accuracy = model.evaluate(X_test, y_test_ohe)
                st.info(f'Model trained, Test accuracy: {accuracy:.2f}')
                fig,ax = plt.subplots(figsize=(8,6))
                ax.set_title('Learning Curve')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss | Accuracy')
                ax.plot(history.history['loss'])
                ax.plot(history.history['accuracy'])
                ax.legend(['Loss', 'Accuracy'])
                st.pyplot(fig)
with tab2:
     uploaded_file = st.file_uploader("Upload Image File",type=['png', 'jpg','jpeg'])
     if uploaded_file is not None:
        col1,col2=st.columns(2) 
        with col1:
            st.image(uploaded_file)
        with col2:
            img = Image.open(uploaded_file)
            img = img.resize((28, 28))
            img = img.convert('L')
            img = np.array(img)
            img[:,:] = [255] - img[:,:]
            img = img.reshape(1, 28, 28)
            img = img / 255
            y = model.predict(img)*100
            y = np.round(y, 2)
            n = np.argsort(y[0])[::-1]
            for i in range(3):
                st.write(labels[n[i]], y[0][n[i]],'%')
