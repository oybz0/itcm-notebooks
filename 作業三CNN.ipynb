{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "235/235 - 3s - loss: 2.1390 - accuracy: 0.1703\n",
      "Epoch 2/50\n",
      "235/235 - 3s - loss: 2.0016 - accuracy: 0.2437\n",
      "Epoch 3/50\n",
      "235/235 - 3s - loss: 1.8091 - accuracy: 0.3679\n",
      "Epoch 4/50\n",
      "235/235 - 2s - loss: 1.5766 - accuracy: 0.4629\n",
      "Epoch 5/50\n",
      "235/235 - 3s - loss: 1.3685 - accuracy: 0.5333\n",
      "Epoch 6/50\n",
      "235/235 - 3s - loss: 1.2049 - accuracy: 0.5932\n",
      "Epoch 7/50\n",
      "235/235 - 3s - loss: 1.0709 - accuracy: 0.6419\n",
      "Epoch 8/50\n",
      "235/235 - 3s - loss: 0.9668 - accuracy: 0.6730\n",
      "Epoch 9/50\n",
      "235/235 - 3s - loss: 0.8900 - accuracy: 0.6971\n",
      "Epoch 10/50\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "import numpy as np\n",
    "import pickle\n",
    "from keras import backend as K\n",
    "from keras import optimizers\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPool2D\n",
    "\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import pickle\n",
    "import keras\n",
    "from keras.layers import LSTM\n",
    "from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten\n",
    "from keras.datasets import mnist\n",
    "from keras.models import Sequential\n",
    "from keras.optimizers import Adam\n",
    "\n",
    "f=open('hw3.pkl','rb')\n",
    "data=pickle.load(f)\n",
    "\n",
    "classes = 9\n",
    "learning_rate = 0.001\n",
    "epochs = 50\n",
    "batch_size = 64\n",
    "optimizer = optimizers.SGD(lr=learning_rate)\n",
    "\n",
    "model = Sequential()\n",
    "model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu', input_shape=(10, 10, 4)))\n",
    "model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation='relu'))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(256, activation='relu'))\n",
    "model.add(Dense(84, activation='relu'))\n",
    "model.add(Dense(9, activation='softmax'))\n",
    "\n",
    "model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])\n",
    "hist = model.fit(x=data['train_gaf'], y=data['train_label_arr'],batch_size=batch_size, epochs=epochs, verbose=2)\n",
    "\n",
    "train_pred = model.predict_classes(data['train_gaf'])\n",
    "test_pred = model.predict_classes(data['test_gaf'])\n",
    "train_label = data['train_label'][:, 0]\n",
    "test_label = data['test_label'][:, 0]\n",
    "\n",
    "train_result_cm = confusion_matrix(train_label, train_pred, labels=range(9))\n",
    "test_result_cm = confusion_matrix(test_label, test_pred, labels=range(9))\n",
    "print(train_result_cm, '\\n'*2, test_result_cm)\n",
    "\n",
    "scores = model.evaluate(data['test_gaf'], data['test_label_arr'], verbose=0)\n",
    "print('CNN test accuracy = ', scores[1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
