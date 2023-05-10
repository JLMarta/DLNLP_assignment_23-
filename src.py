import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #Plotting properties
import seaborn as sns #Plotting properties
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer #Data transformation
from sklearn.model_selection import train_test_split #Data testing
from sklearn.linear_model import LogisticRegression #Prediction Model
from sklearn.metrics import confusion_matrix, accuracy_score #Comparison between real and predicted
from sklearn.preprocessing import LabelEncoder #Variable encoding and decoding for XGBoost
import re #Regular expressions
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
def read_data(path):
  df = pd.read_csv(path, header=None)
  df.columns = ['id','information','type','text']
  #Text transformation
  df["lower"]=df.text.str.lower() #lowercase
  df["lower"]=[str(data) for data in df.lower] #converting all to string
  df["lower"]=df.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex
  data = df
  return data
import pandas as pd

def dataset(datasize):
  train_data = read_data('./Dataset/twitter-entity-sentiment-analysis/twitter_training.csv')
  val_data = read_data('./Dataset/input/twitter-entity-sentiment-analysis/twitter_validation.csv')
  all_data = pd.concat([train_data, val_data], ignore_index=True)
  if datasize == 'All':
    sampled_df = all_data
  elif datasize == 60000:
    # Load the original DataFrame
    df = all_data

    # Separate the 'Irrelevant' category from the rest of the data
    df_irrelevant = df[df['type'] == 'Irrelevant']
    df = df[df['type'] != 'Irrelevant']

    # Group the remaining DataFrame by type
    grouped = df.groupby('type')

    # Create an empty DataFrame to store the sampled rows
    sampled_df = pd.DataFrame()

    # Loop through each group and sample 15,000 rows from each type
    for name, group in grouped:
        sampled_rows = group.sample(15000, random_state=42)
        sampled_df = pd.concat([sampled_df, sampled_rows])

    # Concatenate the sampled rows from all categories into a single DataFrame
    sampled_df = pd.concat([sampled_df, df_irrelevant])

    # Reset the index of the resulting DataFrame
    sampled_df = sampled_df.reset_index(drop=True)
  else:
    # Group the DataFrame by type
    grouped = all_data.groupby('type')
    num_per_type = datasize//4
    # Create an empty DataFrame to store the sampled rows
    sampled_df = pd.DataFrame()

    # Loop through each group and sample 15,000 rows from each type
    for name, group in grouped:
        sampled_rows = group.sample(n=num_per_type, random_state=42)
        sampled_df = pd.concat([sampled_df, sampled_rows])

    # Reset the index of the resulting DataFrame
    sampled_df = sampled_df.reset_index(drop=True)
  type_counts = sampled_df['type'].value_counts()
  type_proportions = type_counts / len(sampled_df)
  print(type_proportions)
  return sampled_df

def bert_data(data):
    # Split the data into training and testing sets
    train_df, rem_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(rem_df, test_size=0.5, random_state=42)
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the text data and convert it to input features
    train_encodings = tokenizer(list(train_df["lower"].astype(str)),
                                truncation=True,
                                padding=True,
                                max_length=128)
    val_encodings = tokenizer(list(val_df["lower"].astype(str)),
                                truncation=True,
                                padding=True,
                                max_length=128)
    test_encodings = tokenizer(list(test_df["lower"].astype(str)),
                               truncation=True,
                               padding=True,
                               max_length=128)
    # encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['type'].values)
    val_labels = label_encoder.fit_transform(val_df['type'].values)
    test_labels = label_encoder.fit_transform(test_df['type'].values)

    # Create TensorFlow datasets from the tokenized data
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels.astype(np.int32)
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels.astype(np.int32)
    ))
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        test_labels.astype(np.int32)
    ))
    print('Data is processed and splitted')
#     tr_type_counts = train_df['type'].value_counts()
#     tr_type_proportions = tr_type_counts / len(train_df)
#     val_type_counts = val_df['type'].value_counts()
#     val_type_proportions = val_type_counts / len(val_df)
#     te_type_counts = test_df['type'].value_counts()
#     te_type_proportions = te_type_counts / len(test_df)
#     print(tr_type_proportions)
#     print(val_type_proportions)
#     print(te_type_proportions)
    return train_dataset, val_dataset, test_dataset, test_labels

def bert_model(train_dataset, val_dataset, test_dataset):
# Load the pre-trained BERT model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    # Compile the model with an appropriate optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train the model on the training data
    history = model.fit(train_dataset.batch(32), epochs=5, 
                        validation_data=val_dataset.batch(32), 
                        batch_size=32)
    return model, history

def train_val_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy 60k data')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss 60k data')
    plt.xlabel('epoch')
    plt.savefig('BERT Train Results 60k data.png')
    print('Train curve for bert model is saved')
def plot_consusion_matrix(model, test_dataset, test_labels):
    # Get predicted labels
    y_pred = np.argmax(model.predict(test_dataset.batch(32)).logits, axis=1)

    # Get true labels
    y_true = test_labels.astype(np.int32)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix as heatmap
    plt.figure(figsize = (8,5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('BERT Confusion Matrix 60k Data')
    plt.savefig('BERT Confusion Matrix 60k Data.jpeg')
    print('Confusion Matrix for bert model is Stored')
    print("Accuracy Score:", accuracy_score(y_true, y_pred))


def data_split(data):
    bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    ngram_range=(1,4))
    train, test = train_test_split(data, test_size=0.1, random_state=42)
    X_train_bow = bow_counts.fit_transform(train.lower)
    X_test_bow = bow_counts.transform(test.lower)
    y_train_bow = train['type']
    y_test_bow = test['type']
    return X_train_bow, y_train_bow, X_test_bow, y_test_bow

def log_reg(X_train_bow, y_train_bow, X_test_bow, y_test_bow ):
    model = LogisticRegression(C=10, solver="liblinear",max_iter=1500,verbose=True)
    # Logistic regression
    model.fit(X_train_bow, y_train_bow)
    # Prediction
    y_pred = model.predict(X_test_bow)
    print("Accuracy: ", accuracy_score(y_test_bow, y_pred) * 100)
    return y_pred

def lr_confusion_matrix(y_test_bow, y_pred):
    cm = confusion_matrix(y_test_bow, y_pred)
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('4-grams LRogistic Regression Confusion Matrix 60k Data')
    plt.savefig('4-grams LRogistic Regression Confusion Matrix 60k Data.jpeg')
    print('Confusion Matrix for LR is Stored')
    corrected_pred_all_labels = np.diag(cm)/y_test_bow.value_counts().sort_index()
    bias = (corrected_pred_all_labels.max() - corrected_pred_all_labels.min())/corrected_pred_all_labels.max()
    print('{:.2%}'.format(bias))


def fourgrams_mlp(X_train_bow, y_train_bow, y_test_bow, y_pred):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                          max_iter=20, early_stopping=True, verbose=True, n_iter_no_change=5)

    # Train the model on the training set, with early stopping based on the validation loss
    history = model.fit(X_train_bow, y_train_bow)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test_bow)
    test_acc = accuracy_score(y_test_bow, y_pred)

    print('Test accuracy:', test_acc)
    return history, y_pred

def fourgrams_mlp_train_val_plot():
    val_acc = history.validation_scores_

    loss = history.loss_curve_

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(val_acc, 'b', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.title('Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, 'r', label='Training Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('4-Grams MLP Train Results 60k data.png')
    print('Train curve for MLP is saved')

def mlp_confusion_matrix(y_test_bow, y_pred):
    cm = confusion_matrix(y_test_bow, y_pred)
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('4-grams MLP Confusion Matrix 60k Data')
    plt.savefig('4-grams MLP Confusion Matrix 60k Data.jpeg')
    corrected_pred_all_labels = np.diag(cm)/y_test_bow.value_counts().sort_index()
    bias = (corrected_pred_all_labels.max() - corrected_pred_all_labels.min())/corrected_pred_all_labels.max()
    print('{:.2%}'.format(bias))
    print('Confusion Matrix for MLP is Stored')





















data= dataset('All')
train_dataset, val_dataset, test_dataset, test_labels= bert_data(data)
model, history = bert_model(train_dataset, val_dataset, test_dataset)
train_val_plot(history)
plot_consusion_matrix(model, test_dataset, test_labels)
X_train_bow, y_train_bow, X_test_bow, y_test_bow = data_split(data)
y_pred = log_reg(X_train_bow, y_train_bow, X_test_bow, y_test_bow)
lr_confusion_matrix(y_test_bow, y_pred)
history = fourgrams_mlp(X_train_bow, y_train_bow, y_test_bow, y_pred)
fourgrams_mlp_train_val_plot()
mlp_confusion_matrix(y_test_bow, y_pred)