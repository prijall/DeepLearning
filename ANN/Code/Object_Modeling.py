from ANN.Code.SoftMax_function import Softmax_Activation
from ANN.Code.Common_loss import Loss_CategoricalCrossentropy
from ANN.Code.Activation_Softmax_Loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
from Layer_Input import Input_Layer

class Model:
    def __init__(self):
        #Creating list of network objects
        self.layers=[]

    #@Adding model to the object:
    def add(self, Layer):
        self.layers.append(Layer)

    #@Adding Loss function and Optimizer for model:
    def set(self, *, Loss, optimizer, accuracy):
        self.Loss=Loss
        self.optimizer=optimizer
        self.accuracy=accuracy

    #@Finalizing the model:
    def finalize(self):
        #Creating and setting input layer
        self.input_layer=Input_Layer()

        #For counting all the objects:
        layer_count=len(self.layers)

        # Initializing list containing trainable layers:
        self.trainable_layers=[]

        #Iterating the object:
        for i in range(layer_count):
            # For first layer, previous layer is input layer:
            if i==0:
                self.layers[i].prev=self.input_layer
                self.layers[i].next=self.layers[i+1]
            
            #this is for all layers except for first and last:
            elif i< layer_count-1:
                self.layers[i].prev=self.layers[i-1]
                self.layers[i].next=self.layers[i+1]

            #for last layer:
            else:
                self.layers[i].prev=self.layers[i-1]
                self.layers[i].next=self.Loss  #next layer is loss
                self.output_layer_activation=self.layers[i]

        if hasattr(self.layers[i], 'weights'):
            self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1],Softmax_Activation) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output=Activation_Softmax_Loss_CategoricalCrossentropy()



    #@Adding train method:
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        self.accuracy.init(y)

        #Main Training Loop:
        for epoch in range(1, epochs+1):
            output=self.forward(X, training=True)

            #Calculate loss:
            data_loss, regularization_loss=self.loss.calculate(output, y)
            loss=data_loss+regularization_loss

            # Get predictions and calculate accuracy:
            predictions=self.output_layer_activation.predictions(output)
            accuracy=self.accuracy.calculate([predictions, y])

            #Performing backward pass:
            self.backward(output, y)

            #Updating Parameters:
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            #printing details:
            if not epoch % print_every:
                print(f'epoch:{epoch}, ' +
                      f'acc:{accuracy:.3f}, '+
                      f'loss:{loss:.3f},'+
                      f'data_loss:{data_loss:.3f},'+
                      f'reg_loss:{regularization_loss:.3f}, '+
                      f'lr:{self.optimizer.current_learning_rate}')
                
            
            if validation_data is not None:
                #for readeability:
                X_val, y_val=validation_data
                output=self.forward(X_val)
                loss=self.loss.calculate(output, y_val)
                
                predictions=self.output_layer_activation.predictions(output)
                accuracy=self.accuracy.calculate(predictions, y_val)

                print(f'validation, '+
                      f'acc:{accuracy:.3f}, '+
                      f'loss:{loss:.3f}')
                     
    #@Forward Pass:
    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output
    
    #@Backward Pass:
    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs=self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
              layer.backward(layer.next.dinputs)
              return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
