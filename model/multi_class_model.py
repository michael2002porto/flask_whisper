import random
import sys

import torch
import torch.nn as nn

import lightning as L

from transformers import BertModel
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall

class MultiClassModel(L.LightningModule):
    def __init__(self,
                 dropout,
                 n_out,
                 lr,
                 hidden_size = 768,
                 model_dim = 768,):
        super(MultiClassModel, self).__init__()

        # save all the hyperparameters
        self.save_hyperparameters()

        # seed untuk weight
        torch.manual_seed(1) # Untuk GPU
        random.seed(1) # Untuk CPU

        # inisialisasi bert
        # sudah di training terhadap dataset tertentu oleh orang di wikipedia
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')

        # hasil dimasukkan ke linear function
        # pre_classifier = agar weight tidak hilang ketika epoch selanjutnya. Agar weight dapat digunakan kembali
        # Disimpan di memori spesifik untuk song lyrics classification
        # di kecilkan dimensinya dari 768 -> 512
        self.pre_classifier = nn.Linear(hidden_size, model_dim)

        self.dropout = nn.Dropout(dropout)

        # n_out = jumlah label
        # jumlah label = 4 (semua usia, anak, remaja, dewasa)
        self.num_classes = n_out
        # output_layer classifier untuk merubah menjadi label
        self.output_layer = nn.Linear(model_dim, self.num_classes)
        #  Activation function / Normalisasi
        self.softmax = nn.Softmax()
                
        # Seberapa dalam rasio si model di optimize
        self.lr = lr
        
        # Persiapan benchmarking
        self.prepare_metrics()

        # menghitung loss function
        self.criterion = nn.BCEWithLogitsLoss()

    # mengambil input dari bert, pre_classifier
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )

        # hidden_state = bert_out[0]
        # pooler = hidden_state[:, 0]
        # Output size (batch size = 20 baris, sequence length = 100 kata / token, hidden_size = 768 tensor jumlah vektor representation dari)
        
        # Full Output Model
        # 12 * 768
        # 12 = layer nya (Filter)
        # 768 = Probabilitas 
        # layer 12

        # dimensi pooler output = 1 * 768
        bert_out = bert_out.pooler_output   #ambil output layer terakhir
        out = self.dropout(bert_out)  #menghilangkan memory

        # pre classifier untuk mentransfer wight output ke epch selanjuntya
        out = self.pre_classifier(out)     #pindah ke memori khusus klasifikasi
        
        # kontrol hasil pooler min -1 max 1
        # pooler = torch.nn.Tanh()(pooler)
        
        # 0.02312312412413131 -> 0.023412 (normalisasi) -> 0 -> 1
        # -0.3124211 -> 0.00012
        out = self.output_layer(out)    # output_layer classifier untuk memprojeksikan hasil pooler (768) ke jumlah label (4)
        out = self.softmax(out)     #menstabilkan sehingga 0 - 1

        # pooler = self.dropout(pooler)

        return out

    def prepare_metrics(self):
        task = "multiclass"
        
        self.acc_metrics = Accuracy(task = task, num_classes = self.num_classes)
        
        self.f1_metrics_micro = F1Score(task = task, num_classes = self.num_classes, average = "micro")
        self.f1_metrics_macro = F1Score(task = task, num_classes = self.num_classes, average = "macro")
        self.f1_metrics_weighted = F1Score(task = task, num_classes = self.num_classes, average = "weighted")
        
        self.prec_metrics_micro = Precision(task = task, num_classes = self.num_classes, average = "micro")
        self.prec_metrics_macro = Precision(task = task, num_classes = self.num_classes, average = "macro")
        self.prec_metrics_weighted = Precision(task = task, num_classes = self.num_classes, average = "weighted")
        
        self.recall_metrics_micro = Recall(task = task, num_classes = self.num_classes, average = "micro")
        self.recall_metrics_macro = Recall(task = task, num_classes = self.num_classes, average = "macro")
        self.recall_metrics_weighted = Recall(task = task, num_classes = self.num_classes, average = "weighted")

        # to make use of all the outputs
        self.training_step_output = []
        self.validation_step_output = []
        self.test_step_output = []
        
    def benchmarking_step(self, pred, target):
        '''
        output pred / target = 
        [
            [0.001, 0.80],
            [0.8, 0.0001],
            [0.8, 0.0001],
            [0.8, 0.0001],
            [0.8, 0.0001]
        ]
        
        y_pred -> [1, 0, 0, 0, 0]
        '''
        
        pred = torch.argmax(pred, dim = 1)
        target = torch.argmax(target, dim = 1)
        
        metrics = {}
        metrics["accuracy"] = self.acc_metrics(pred, target)
        metrics["f1_micro"] = self.f1_metrics_micro(pred, target)
        metrics["f1_macro"] = self.f1_metrics_macro(pred, target)
        metrics["f1_weighted"] = self.f1_metrics_weighted(pred, target)
        metrics["prec_micro"] = self.prec_metrics_micro(pred, target)
        metrics["prec_macro"] = self.prec_metrics_macro(pred, target)
        metrics["prec_weighted"] = self.prec_metrics_weighted(pred, target)
        metrics["recall_micro"] = self.recall_metrics_micro(pred, target)
        metrics["recall_macro"] = self.recall_metrics_macro(pred, target)
        metrics["recall_weighted"] = self.recall_metrics_weighted(pred, target)
        
        return metrics

    def configure_optimizers(self):
        # di dalam parameter adam, parameters untuk mengambil kesuluruhan input yg di atas

        # Fungsi adam 
        # Tranfer epoch 1 ke epoch 2
        # Mengontrol (efisiensi) loss
        # Proses training lebih cepat
        # Tidak memakan memori berlebih
        #Learning rate semakin tinggi maka hasil itunya semakin besar
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)   #untuk menjaga training model improve
        return optimizer

    def training_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        # Ke tiga parameter di input dan di olah oleh method / function forward()
        y_pred = self(
            input_ids = x_input_ids,
            attention_mask = x_attention_mask,
            token_type_ids = x_token_type_ids
        )

        #y_pred semakin salah, maka semakin tinggi loss
        loss = self.criterion(y_pred, target = y.float())
        
        metrics = self.benchmarking_step(pred = y_pred, target = y)     #tahu skor
        metrics["loss"] = loss
        metrics_loss = loss
        
        self.training_step_output.append(metrics)
        self.log_dict({"train_loss": metrics_loss}, prog_bar = True, on_epoch = True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        # Ke tiga parameter di input dan di olah oleh method / function forward()
        y_pred = self(
            input_ids = x_input_ids,
            attention_mask = x_attention_mask,
            token_type_ids = x_token_type_ids
        )

        #y_pred semakin salah, maka semakin tinggi loss
        loss = self.criterion(y_pred, target = y.float())
        
        metrics = self.benchmarking_step(pred = y_pred, target = y)     #tahu skor
        metrics["loss"] = loss
        metrics_loss = loss
        
        self.validation_step_output.append(metrics)
        self.log_dict({"val_loss": metrics_loss}, prog_bar = True, on_epoch = True)
        
        return loss

    def test_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        # Ke tiga parameter di input dan di olah oleh method / function forward()
        y_pred = self(
            input_ids = x_input_ids,
            attention_mask = x_attention_mask,
            token_type_ids = x_token_type_ids
        )

        #y_pred semakin salah, maka semakin tinggi loss
        loss = self.criterion(y_pred, target = y.float())
        
        metrics = self.benchmarking_step(pred = y_pred, target = y)     #tahu skor
        metrics["loss"] = loss
        
        self.test_step_output.append(metrics)
        self.log_dict(metrics, prog_bar = True, on_epoch = True)
        
        return loss

    # def predict_step(self, batch, batch_idx):
    #     # Tidak ada transfer weight
    #     x_input_ids, x_token_type_ids, x_attention_mask, y = batch

    #     out = self(input_ids = x_input_ids,
    #                attention_mask = x_attention_mask,
    #                token_type_ids = x_token_type_ids)
    #     # Ke tiga parameter di input dan di olah oleh method / function forward

    #     pred = out.argmax(1).cpu()
    #     true = y.argmax(1).cpu()

    #     outputs = {"predictions": out, "labels": y}
    #     self.predict_step_outputs.append(outputs)

    #     # return [pred, true]
    #     return outputs

    # def on_train_epoch_end(self):
    #     labels = []
    #     predictions = []

    #     for output in self.training_step_outputs:
    #         for out_lbl in output["labels"].detach().cpu():
    #             labels.append(out_lbl)
    #         for out_pred in output["predictions"].detach().cpu():
    #             predictions.append(out_pred)

    #     # argmax(dim=1) = convert one-hot encoded labels to class indices
    #     labels = torch.stack(labels).int().argmax(dim=1)
    #     predictions = torch.stack(predictions).argmax(dim=1)

    #     print("\n")
    #     print("labels = ", labels)
    #     print("predictions = ", predictions)
    #     print("num_classes = ", self.num_classes)

    #     # Hitung akurasi
    #     accuracy = Accuracy(task = "multiclass", num_classes = self.num_classes)
    #     acc = accuracy(predictions, labels)

    #     # Print Akurasinya
    #     print("Overall Training Accuracy : ", acc)
    #     print("\n")
    #     # sys.exit()

    #     # free memory
    #     self.training_step_outputs.clear()

    # def on_predict_epoch_end(self):
    #     labels = []
    #     predictions = []

    #     for output in self.predict_step_outputs:
    #         # print(output[0]["predictions"][0])
    #         # print(len(output))
    #         # break
    #         for out_lbl in output["labels"].detach().cpu():
    #             labels.append(out_lbl)
    #         for out_pred in output["predictions"].detach().cpu():
    #             predictions.append(out_pred)

    #     # argmax(dim=1) = convert one-hot encoded labels to class indices
    #     labels = torch.stack(labels).int().argmax(dim=1)
    #     predictions = torch.stack(predictions).argmax(dim=1)

    #     print("\n")
    #     print("labels = ", labels)
    #     print("predictions = ", predictions)
    #     print("num_classes = ", self.num_classes)

    #     accuracy = Accuracy(task = "multiclass", num_classes = self.num_classes)
    #     acc = accuracy(predictions, labels)
    #     print("Overall Testing Accuracy : ", acc)
    #     print("\n")
    #     # sys.exit()

    #     # free memory
    #     self.predict_step_outputs.clear()