from My_Neural_Network import *
def main():
    #Load MNIST dataset
    X_train,Y_train,X_val,Y_val = get_MNIST_data()
    #Define network architecture
    layers = [Linear(784,128),
              Batch_Norm(128),
              Activation(ReLU),
              Linear(128,256),
              Batch_Norm(256),
              Activation(ReLU),
              Linear(256,10),
              Activation(softmax)]
              #Example : Dropout(0.3) here not necessary
    #Initialize network
    My_Net = Net(layers,"NLL",metrics = ["per_class_accuracy","accuracy","mistake_matrix"]
                 ,plot = True,lr_rate = 0.001)
    #Train the model
    My_Net.train(X_train,Y_train,epoc = 20,X_val = X_val,Y_val = Y_val)
    #Evaluate performance
    print(My_Net.evaluate(X_val,Y_val))
if __name__ == "__main__":
    main()