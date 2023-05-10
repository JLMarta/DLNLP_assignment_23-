from src import dataset, bert_data, bert_model, train_val_plot, plot_consusion_matrix, data_split, log_reg, lr_confusion_matrix, fourgrams_mlp, fourgrams_mlp_train_val_plot, mlp_confusion_matrix
# from src import bias_data, bias_bert_model, bias_train_val_plot, bias_plot_consusion_matrix, bias_data_split,bias_log_reg, bias_lr_confusion_matrix, bias_fourgrams_mlp_train_val_plot, bias_fourgrams_mlp, bias_mlp_confusion_matrix
# Performance on Normal Distribution of data
# Only performance on normal data will be run for demonstration
data= dataset('All')
train_dataset, val_dataset, test_dataset, test_labels= bert_data(data)
model, history = bert_model(train_dataset, val_dataset, test_dataset)
train_val_plot(history)
plot_consusion_matrix(model, test_dataset, test_labels)
X_train_bow, y_train_bow, X_test_bow, y_test_bow = data_split(data)
y_pred = log_reg(X_train_bow, y_train_bow, X_test_bow, y_test_bow)
lr_confusion_matrix(y_test_bow, y_pred)
history = fourgrams_mlp(X_train_bow, y_train_bow, y_test_bow, X_test_bow)
fourgrams_mlp_train_val_plot(history)
mlp_confusion_matrix(y_test_bow, y_pred)

# Performance on biased data
# Uncomment line 2 and belowing part to run experiment on biased data
# train_dataset, val_dataset, test_dataset, test_labels = bias_data(data)
# model, history = bias_bert_model(train_dataset, val_dataset, test_dataset)
# bias_train_val_plot(history)
# bias_plot_consusion_matrix(model, test_dataset, test_labels)
# X_train_bow, y_train_bow, X_test_bow, y_test_bow = bias_data_split(data)
# y_pred = bias_log_reg(X_train_bow, y_train_bow, X_test_bow, y_test_bow )
# bias_lr_confusion_matrix(y_test_bow, y_pred)
# history,y_pred = bias_fourgrams_mlp(X_train_bow, y_train_bow, y_test_bow, X_test_bow)
# bias_fourgrams_mlp_train_val_plot()
# bias_mlp_confusion_matrix(y_test_bow, y_pred)