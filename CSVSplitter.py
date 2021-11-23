if __name__ == '__main__':

    images = "ORIGINAL_train_image.csv"
    labels = "ORIGINAL_train_label.csv"

    counter = 0
    group = 0
    output_image_file = None
    output_label_file = None

    with open(images) as image_file, open(labels) as label_file:
        for lineImg, lineLbl in zip(image_file, label_file):
            if counter % 10000 == 0:
                counter += 1
                group += 1
                if not output_image_file is None and not output_label_file is None:
                    output_image_file.close()
                    output_label_file.close()

                output_image_file_name = "train_image_" + str(group) + ".csv"
                output_label_file_name = "train_label_" + str(group) + ".csv"

                output_image_file = open(output_image_file_name, "w+")
                output_image_file.write(lineImg)
                output_label_file = open(output_label_file_name, "w+")
                output_label_file.write(lineLbl)
            else:
                counter += 1
                output_image_file.write(lineImg)
                output_label_file.write(lineLbl)
