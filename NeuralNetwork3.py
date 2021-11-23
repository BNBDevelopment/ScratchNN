import datetime
import sys

import numpy

from ClassDefinitions import NeuralNet


def trainModelOnInput(nnModel, numberOfEpochs):
    input_arguments = sys.argv
    train_image_file_path = input_arguments[1]
    train_label_file_path = input_arguments[2]

    successes = 0
    counter = 1

    for epoch_number in range(0, numberOfEpochs):
        print("Starting Epoch: " + str(epoch_number + 1))
        with open(train_image_file_path) as train_image_file, open(train_label_file_path) as train_label_file:
            for lineImg, lineLbl in zip(train_image_file, train_label_file):
                #print("lineImg: " + str(lineImg))
                #print("lineLbl: " + str(lineLbl))
                input_img_values = numpy.array(lineImg.split(","), dtype='float64')

                #print("Iteration number: " + str(counter) + "\t Start Time: " + str(datetime.datetime.now()))
                counter += 1

                #Forward Pass
                #print("Forward Start: " + str(datetime.datetime.now()))
                prediction_values = nnModel.pushInputThroughModel(input_img_values)
                #print("Forward End: " + str(datetime.datetime.now()))

                #Back propogate error
                #print("EXPECTED VALUE: " + str(lineLbl))
                #print("Backward Start: " + str(datetime.datetime.now()))
                nnModel.backPropogateThroughLayers(lineLbl)
                #print("Backward End: " + str(datetime.datetime.now()))



def processTestInput(nnModel):
    input_arguments = sys.argv
    test_image_file_path = input_arguments[3]


    if len(input_arguments) == 5:
        test_label_file_path = input_arguments[4]
        correct_count = 0
        total_count = 0

        with open(test_image_file_path) as test_image_file, open(test_label_file_path) as test_label_file:
            for lineImg, lineLbl in zip(test_image_file, test_label_file):
                input_img_values = numpy.array(lineImg.split(","), dtype='float64')
                total_count += 1

                # Forward Pass
                prediction_values = nnModel.pushInputThroughModel(input_img_values)
                max_value_index = prediction_values.argmax(axis=0)

                if int(max_value_index) == int(lineLbl):
                    correct_count += 1
            test_image_file.close()
            test_label_file.close()

        print("TEST CORRECTNESS: " + str((correct_count/total_count) * 100) + "%")

    else:
        output_file_path = "test_predictions.csv"

        with open(output_file_path, 'w+') as output_file:
            with open(test_image_file_path) as test_image_file:
                for lineImg in test_image_file:
                    input_img_values = numpy.array(lineImg.split(","), dtype='float64')

                    # Forward Pass
                    prediction_values = nnModel.pushInputThroughModel(input_img_values)
                    max_value_index = prediction_values.argmax(axis=0)
                    output_file.write(str(max_value_index) + "\n")
            test_image_file.close()
        output_file.close()



def createTestOutput():
    output_file_path = "test_predictions.csv"
    pass



if __name__ == '__main__':

    #28x28 = 784
    print("Initializing... Start Time: " + str(datetime.datetime.now()))
    #nnModel = NeuralNet(784, 10, 3, [392,196,98])
    nnModel = NeuralNet(784, 10, 1, [128])
    nnModel.setAlpha(0.01)
    print("Initializing Complete! End Time: " + str(datetime.datetime.now()))



    print("Training Model... Start Time: " + str(datetime.datetime.now()))
    number_of_epochs = 12
    trainModelOnInput(nnModel, number_of_epochs)
    print("Training Complete! End Time: " + str(datetime.datetime.now()))


    print("Testing Model... Start Time: " + str(datetime.datetime.now()))
    processTestInput(nnModel)
    print("Testing Complete! End Time: " + str(datetime.datetime.now()))