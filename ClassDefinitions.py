import datetime

import numpy
import math

class NeuralNetHiddenLayer():
    def __init__(self, layerNumber, numberOfNodes):
        self.layer_number = layerNumber
        self.number_of_nodes = numberOfNodes
        self.alpha = .1

        self.activation_function = SigmoidActivationFunction()
        self.next_layer = None
        self.previous_layer = None
        self.calculated_values = numpy.zeros((numberOfNodes))
        self.deltas = numpy.zeros((numberOfNodes))

        #print("numberOfNodes: " + str(numberOfNodes))
        self.input_sums_array = numpy.zeros((numberOfNodes)) #n rows, 1 column
        self.activation_function_array = numpy.ones((numberOfNodes)) #list of w values, n rows 1 column, initialized as all 1

        self.output_links_array = numpy.empty((0,3)) #origin node index, destination node index, weight on the link (#, w)

    def getAlpha(self,):
        return self.alpha
    def setAlpha(self, new_alpha):
        self.alpha = new_alpha

    def getNextLayer(self):
        return self.next_layer
    def setNextLayer(self, newNextNNLayer):
        #Setting the next layer and initializing the connections to random weights
        self.next_layer = newNextNNLayer

        next_layer_number = self.next_layer.layer_number
        for currentLayerNodeNumber in range(0, self.number_of_nodes):
            #print("currentLayerNodeNumber: " + str(currentLayerNodeNumber))
            for nextLayerNodeNumber in range(0, self.next_layer.number_of_nodes):
                random_initial_weight = numpy.random.uniform(-.1, .1)

                self.output_links_array = numpy.vstack((self.output_links_array, [currentLayerNodeNumber, nextLayerNodeNumber, random_initial_weight]))

    def getPreviousLayer(self):
        return self.previous_layer
    def setPreviousLayer(self, prevLayer):
        self.previous_layer = prevLayer


    def pushLayerValuesForward(self):
        #when pushing new values from current layer to next layer, we need to reset next layer sums to 0 to start
        #self.next_layer.input_sums_array = numpy.zeros(self.next_layer.number_of_nodes)

        #Now update the next layer sums given input values/sums on this layer

        vectorized_sigmoid_function = numpy.vectorize(self.activation_function.sigmoid_function)
        self.calculated_values = vectorized_sigmoid_function(self.input_sums_array)

        #for next_layer_node_index in range(0, self.next_layer.number_of_nodes):
        #    matching_indexes = numpy.where(self.output_links_array[:, 1] == next_layer_node_index)[0]

        #    weights = self.output_links_array[matching_indexes][:,2]
        #    calculated_values = numpy.multiply(weights, self.calculated_values)

        #    self.next_layer.input_sums_array[next_layer_node_index] = numpy.sum(calculated_values)


        stretched_calculated_values = numpy.repeat(self.calculated_values, self.next_layer.number_of_nodes, axis=0)
        output_values = stretched_calculated_values * self.output_links_array[:,2]
        for next_layer_node_index in range(0, self.next_layer.number_of_nodes):
            indices_to_sum = numpy.arange(next_layer_node_index, output_values.shape[0], self.next_layer.number_of_nodes)
            self.next_layer.input_sums_array[next_layer_node_index] = numpy.sum(output_values[indices_to_sum])


    def g_prime(self, inputVal):
        g_input = self.activation_function.sigmoid_function(inputVal)
        return g_input * (1 - g_input)

    def backPropogate(self, expected_output_value):
        #Output layer
        if self.next_layer is None:
            for node_index in range(0, self.number_of_nodes):
                self.deltas[node_index] = self.g_prime(self.input_sums_array[node_index]) * (expected_output_value[node_index] - self.calculated_values[node_index])
        else:
            for node_index in range(0, self.number_of_nodes):
                current_node_connections_to_next_layer = numpy.where(self.output_links_array[:, 0] == node_index)[0]

                summation_weights_deltas = 0
                for current_layer_node in current_node_connections_to_next_layer:
                    connected_to_node_index = self.output_links_array[current_layer_node, 1]
                    delta_connected_to_node = self.next_layer.deltas[int(connected_to_node_index)]
                    weight_of_connection = self.output_links_array[current_layer_node, 2]
                    summation_weights_deltas = summation_weights_deltas + (weight_of_connection * delta_connected_to_node)

                self.deltas[node_index] = self.g_prime(self.input_sums_array[node_index]) * summation_weights_deltas

            #now that we have deltas can update weights

            #TODO: Test if this works - UPDATING WEIGHTS
            #self.input_sums_array[:, 2] = self.input_sums_array[:, 2]  + (self.alpha * self.calculated_values[ self.input_sums_array[:, 0] ] * self.next_layer.deltas[ self.input_sums_array[:, 1] ])

            for connection_index in range(0,self.output_links_array.shape[0]):
                connection = self.output_links_array[connection_index]
                origin_node_index = connection[0]
                destination_node_index = connection[1]
                destination_delta = self.next_layer.deltas[int(destination_node_index)]

                self.output_links_array[connection_index,2] = connection[2] + (self.alpha * self.calculated_values[int(origin_node_index)] * destination_delta)


    def recieveInputLayerValues(self, inputLayerValues):
        self.input_sums_array = inputLayerValues

    def returnOutputValues(self):
        vectorized_sigmoid_function = numpy.vectorize(self.activation_function.sigmoid_function)
        self.calculated_values = vectorized_sigmoid_function(self.input_sums_array)

        #for currentLayerNodeNumber in range(0, self.number_of_nodes):
        #print("self.input_sums_array("+ str(self.input_sums_array))
            #sum_of_inputs = self.input_sums_array[currentLayerNodeNumber]
            #activation_value = self.activation_function.sigmoid_function(sum_of_inputs)
            #self.calculated_values[currentLayerNodeNumber] = activation_value

            #print("Output Numbprint("+ "\tOutput Value: " + str(self.calculated_values[currentLayerNodeNumber]))
        #max_value_index = self.input_sums_array.argmax(axis=0)
        #print("SELECTED: " + str(max_value_index))
        return self.input_sums_array

class NeuralNet():
    input_layer = None
    output_layer = None
    hidden_layers = []

    def setAlpha(self, param):
        print("Alpha: " + str(param))
        self.input_layer.setAlpha(param)
        self.output_layer.setAlpha(param)
        for layer in self.hidden_layers:
            layer.setAlpha(param)


    def __init__(self, numberInputNodes, numberOutputNodes, numberHiddenLayers, hiddenLayerSizes):
        print("numberInputNodes: " + str(numberInputNodes) + "\tnumberOutputNodes: " + str(numberOutputNodes))
        print("numberHiddenLayers: " + str(numberHiddenLayers) + "\thiddenLayerSizes: " + str(hiddenLayerSizes))

        self.input_layer = NeuralNetHiddenLayer(0, numberInputNodes)
        self.output_layer = NeuralNetHiddenLayer(numberHiddenLayers+1, numberOutputNodes)

        #previous layer pass
        for i in range(0, numberHiddenLayers):
            new_hidden_layer = NeuralNetHiddenLayer(i + 1, hiddenLayerSizes[i])
            if i == 0:
                new_hidden_layer.setPreviousLayer(self.input_layer)
            else:
                new_hidden_layer.setPreviousLayer(self.hidden_layers[i - 1])

            self.hidden_layers.append(new_hidden_layer)
        self.output_layer.setPreviousLayer(self.hidden_layers[numberHiddenLayers-1])

        #next layer pass
        for i in range(0, numberHiddenLayers):
            hidden_layer = self.hidden_layers[i]
            if i == numberHiddenLayers-1:
                hidden_layer.setNextLayer(self.output_layer)
            else:
                hidden_layer.setNextLayer(self.hidden_layers[i + 1])


        self.input_layer.setNextLayer(self.hidden_layers[0])

        print("Neural Net Initialization Complete.")


    def backPropogateThroughLayers(self, outputLabel):
        #TODO
        #print("BP Start Time: " + str(datetime.datetime.now()))
        expected_output_values = numpy.zeros((10))
        expected_output_values[int(outputLabel)] = 3

        workingLayer = self.output_layer
        while not workingLayer.previous_layer is None:
            workingLayer.backPropogate(expected_output_values)
            workingLayer = workingLayer.previous_layer

        #print("BP End Time: " + str(datetime.datetime.now()))

    def pushInputThroughModel(self, vectorOfInputImageValues):
        # TODO
        #print("PUSH Start Time: " + str(datetime.datetime.now()))
        self.input_layer.recieveInputLayerValues(vectorOfInputImageValues)
        self.input_layer.pushLayerValuesForward()

        for hiddenLayer in self.hidden_layers:
            hiddenLayer.pushLayerValuesForward()

        #print("PUSH End Time: " + str(datetime.datetime.now()))
        return self.output_layer.returnOutputValues()




class SigmoidActivationFunction():
    x = 0 #input
    w = 1
    e = math.e

    def getW(self):
        return self.w
    def setW(self, new_w_val):
        self.w = new_w_val

    def sigmoid_function(self, x):
        logistic_w_x = 1 / (1 + (self.e ** (-self.w * x)))
        return logistic_w_x