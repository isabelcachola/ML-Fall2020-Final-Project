# Torch imports
import torch
import torch.nn.functional as F

# BiLSTM imports
import io
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, classification_report, log_loss
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Flatten
from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Log Regression imports
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import pickle

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 8)

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        out1 = self.linear1(x)
        activated = F.relu(out1)
        logits = self.linear2(activated)

        return logits

# This is Olga's code in class form
class LogisticRegression:
    def __init__(self, load_model_path=None):
        if load_model_path is not None:
            with open(load_model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            self.model = linear_model.LogisticRegression
        
    
    def train(self, X_train, y_train, X_dev, y_dev):
        cws = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        self.model = self.model(class_weight=cws, random_state=1, max_iter=500)

        self.model.fit(X_train, y_train)
        
        y_hat = self.model.predict_proba(X_dev)
        print('Validation Loss:', log_loss(y_dev, y_hat))
        print('Validation Accuracy', (y_hat.argmax(axis = 1) == y_dev).mean())
        print('Validation F1 Score:', f1_score(y_dev, y_hat.argmax(axis = 1), average='weighted'))

        with open("models/logreg.pkl", 'wb') as file:
            pickle.dump(self.model, file)
    
    def test(self, X_test, y_test=None):
        predictions = self.model.predict(X_test)
        np.savetxt("preds/logreg-preds.txt", predictions, fmt='%d')
        return predictions

# This is Kevin's code in class form
class BiLSTM:
    def __init__(self, epochs=5, 
            batch_size=36, 
            max_seq_len=25,
            fit_verbose=2,
            print_summary=True,
            load_model_path=None,
            tokenizer_path=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len 
        self.fit_verbose = fit_verbose
        self.print_summary = print_summary
        self.encoder = LabelEncoder()

        if load_model_path:
            self.model = load_model(load_model_path)
            with open(tokenizer_path) as f:
                data = json.load(f)
                self.tokenizer = tokenizer_from_json(data)
        else:
            self.model = self.model_1b 
            self.tokenizer = Tokenizer()
        
    def train(self, X_train, y_train, X_dev, y_dev):
        self.tokenizer.fit_on_texts(X_train)

        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_train = pad_sequences(X_train, maxlen=self.max_seq_len)

        X_dev = self.tokenizer.texts_to_sequences(X_dev)
        X_dev = pad_sequences(X_dev, maxlen=self.max_seq_len)

        y_train = self.encoder.fit_transform(y_train)
        y_train = to_categorical(y_train) 

        y_dev = self.encoder.fit_transform(y_dev)
        y_dev= to_categorical(y_dev) 

        m = self.model()

        y_train_int = np.argmax(y_train,axis=1)
        cws = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)

        if self.print_summary:
            print(m.summary())
        m.fit(
            X_train, 
            y_train, 
            validation_data=(X_dev, y_dev),  
            epochs=self.epochs, 
            batch_size=self.batch_size,
            verbose=self.fit_verbose
        ) 
        predictions = m.predict(X_dev, verbose=1)
        print('Validation Loss:', log_loss(y_dev, predictions))
        print('Validation Accuracy', (predictions.argmax(axis = 1) == y_dev.argmax(axis = 1)).mean())
        print('Validation F1 Score:', f1_score(y_dev.argmax(axis = 1), predictions.argmax(axis = 1), average='weighted'))
        m.save('models/bilstm.keras')
        tokenizer_json = self.tokenizer.to_json()
        with io.open('models/bilstm-tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        self.model = m

    def model_1b(self):
        """
        Using a Bidiretional LSTM. 
        """
        model = Sequential()
        model.add(Embedding(input_dim=(len(self.tokenizer.word_counts) + 1), output_dim=128, input_length=self.max_seq_len))
        model.add(SpatialDropout1D(0.3))
        model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def test(self, X_test, y_test=None):
        X_test = self.tokenizer.texts_to_sequences(X_test)
        X_test = pad_sequences(X_test, maxlen=self.max_seq_len)
        
        predictions = self.model.predict(X_test, verbose=1)
        if y_test is not None:
            y_test = self.encoder.fit_transform(y_test)
            y_test = to_categorical(y_test)     
            print('Test Loss:', log_loss(y_test, predictions))
            print('Test Accuracy', (predictions.argmax(axis = 1) == y_test.argmax(axis = 1)).mean())
            print('Test F1 Score:', f1_score(y_test.argmax(axis = 1), predictions.argmax(axis = 1), average='weighted'))
        predictions = np.argmax(predictions, axis=1)
        np.savetxt("preds/bilstm-preds.txt", predictions, fmt='%d')
        return predictions