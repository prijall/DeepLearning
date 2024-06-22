from Layer_Input import Input_Layer

class Model:
    def __init__(self):
        #Creating list of network objects
        self.layers=[]

    #@Adding model to the object:
    def add(self, Layer):
        self.layers.append(Layer)

    #@Adding Loss function and Optimizer for model:
    def set(self, *, Loss, optimizer):
        self.Loss=Loss
        self.optimizer=optimizer

    #@Finalizing the model:
    def finalize(self):
        #Creating and setting input layer
        self.input_layer=Input_Layer()

        #For counting all the objects:
        layer_count=len(self.layers)

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

    #@Adding train method:
    def train(self, X, y, *, epochs=1, print_every=1):
        #Main Training Loop:
        for epoch in range(1, epochs+1):
            output=self.forward(X)

    #@Forward Pass:
    def forward(self, X):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        return layer.output
    

