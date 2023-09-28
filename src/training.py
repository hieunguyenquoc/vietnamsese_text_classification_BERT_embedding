import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from preprocess import Preprocess
from model import TextClassification
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from parser_param import parameter_parser
SEED = 2019
torch.manual_seed(SEED)

# class DataMapper(Dataset):
#     def __init__(self,x ,y):
#         self.x = x
#         self.y = y
    
#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, index):
#         return self.x[index], self.y[index]    

class Execute:
    def __init__(self,args):
        self.__init_data__(args)
        if torch.cuda.is_available():
            self.device = "cuda:0"
            print("Run on GPU")
        else:
            self.device = "cpu"
            print("Run on CPU")
        self.batch_size = args.batch_size
        self.model = TextClassification(args)
        self.model.to(self.device)

    def __init_data__(self, args):
        self.preprocess = Preprocess(args)
        self.preprocess.load_data()
        self.preprocess.Tokenize()

        self.raw_train_data = self.preprocess.X_train
        self.raw_test_data = self.preprocess.X_test

        # self.x_train = self.preprocess.sequence_to_text(self.raw_train_data)
        self.x_test = self.preprocess.sequence_to_text(self.raw_test_data)

        self.y_train = self.preprocess.Y_train
        self.y_test = self.preprocess.Y_test

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        loss_function = nn.CrossEntropyLoss(reduction="sum")
        loss_function = loss_function.to(self.device)
        train_loss = []
        test_loss = []
        

        print(self.model)
        for epoch in range(args.epochs):
            
            self.model.train()
            avg_loss = 0
            for i in range(0, len(self.raw_train_data), args.batch_size):
                
                batch = self.raw_train_data[i:i+args.batch_size]

                y_pred = self.model(batch)
                loss = loss_function(y_pred, torch.from_numpy(self.y_train[i:i+args.batch_size]).to(self.device))

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                avg_loss += loss.item() / len(batch)

        # test_prediction = self.evaluation()

        # train_accuracy = self.calculate(self.y_train, prediction)

        # test_accuracy = self.calculate(self.y_test, test_prediction)

        # print("Epoch : %.5f, loss : %.5f, Train Accuracy : %.5f, Test Accuracy : %.5f" % (epoch +1, loss.item(), train_accuracy, test_accuracy))
            self.model.eval()
            avg_test_loss = 0
            test_preds = np.zeros((len(self.x_test), len(self.preprocess.label_encoder.classes_)))
            with torch.no_grad():
                for k,j in enumerate(range(0, len(self.raw_test_data), args.batch_size)):
    
                    batch = self.raw_test_data[j:j+args.batch_size]
             
                    y_pred = self.model(batch)
                  
                    avg_test_loss += loss_function(y_pred, torch.from_numpy(self.y_test[j:j+args.batch_size]).to(self.device)).item() / len(batch)
                
                    test_preds[k * args.batch_size:(k+1) * args.batch_size] = F.softmax(y_pred.cpu()).numpy() 

                #Check accuracy
                test_accuracy = sum(test_preds.argmax(axis=1)==self.preprocess.Y_test)/len(self.preprocess.Y_test)
                train_loss.append(avg_loss)
                test_loss.append(avg_test_loss)
                print('Epoch {}/{} \t loss={:.4f} \t test_loss={:.4f}  \t test_acc={:.4f}'.format(
                epoch + 1, args.epochs, avg_loss, avg_test_loss, test_accuracy))
        
        torch.save(self.model, "model/viet_nam_classification.ckpt")
        print("Done train phase")
if __name__ == "__main__":
    args = parameter_parser()
    execute = Execute(args)
    execute.train()
