from NaiveBayesClassifier import NaiveBayesClassifier
from PerceptronClassifier import PerceptronClassifier

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

def readImagesData(file_name,):
    file_input = [l[:-1] for l in open(file_name).readlines()]
    file_input.reverse()
    items = []
    for i in range(len(file_input) // IMAGE_HEIGHT):
        data = []
        for j in range(IMAGE_WIDTH):
            data.append(list(file_input.pop()))
        items.append(data)
    return items

def readLabels(filename, n):
  """
  Reads n labels from a file and returns a list of integers.
  """
  fin = [l[:-1] for l in open(filename).readlines()]
  labels = []
  for line in fin[:min(n, len(fin))]:
    if line == '':
        break
    labels.append(int(line))
  return labels


if __name__=="__main__":

    train_data = readImagesData("./digitdata/trainingimages")
    validation_data = readImagesData("./digitdata/validationimages")
    test_data = readImagesData("./digitdata/testimages")

    train_labels = readLabels("./digitdata/traininglabels" , len(train_data))
    validation_labels = readLabels("./digitdata/validationlabels", len(validation_data))
    test_labels = readLabels("./digitdata/testlabels", len(test_data))

    print("----------------------------- Question 1 -----------------------------")
    naiveBayesClassifier = NaiveBayesClassifier()
    print("\n----------------------------- Section 1 -----------------------------")
    
    for alpha in range(1,6):
        print("\nTraining with k-smooth=" + str(alpha))
        naiveBayesClassifier.train(train_data, train_labels, alpha)

        print("Predicting on the validation set..")
        accuracy = naiveBayesClassifier.predict(validation_data, validation_labels)
        print("The accuracy is: " + str(accuracy * 100) + "%")
    
    print("\n----------------------------- Section 2 -----------------------------")
    alpha = 1
    print("\nTraining with k-smooth=" + str(alpha))
    naiveBayesClassifier.train(train_data, train_labels, alpha)

    print("Predicting on the tests set..")
    accuracy = naiveBayesClassifier.predict(test_data, test_labels)
    print("The accuracy is: " + str(accuracy * 100) + "%")

    print("\n----------------------------- Section 3 -----------------------------")
    print("\nPredicting on the training set..")
    accuracy = naiveBayesClassifier.predict(train_data, train_labels)
    print("The accuracy is: " + str(accuracy * 100) + "%")

    print("\n\n----------------------------- Question 2 -----------------------------")
    perceptronClassifier = PerceptronClassifier()

    print("\n----------------------------- Section 1 -----------------------------")
    epochs_number = 3
    perceptronClassifier.train(train_data, train_labels, epochs_number)
    
    accuracy_train_data = perceptronClassifier.predict(train_data, train_labels)
    print("\nPrediction accuracy on train data is: " + str(accuracy_train_data * 100) + "%")

    accuracy_test_data = perceptronClassifier.predict(test_data, test_labels)
    print("Prediction accuracy on test data is: " + str(accuracy_test_data * 100) + "%")
    

    print("\n----------------------------- Section 2 -----------------------------")

    for epochs_number in range(1,6):
        print("\nTraining with " + str(epochs_number) + " epochs..")
        perceptronClassifier.train(train_data, train_labels, epochs_number)

        accuracy_train_data = perceptronClassifier.predict(train_data, train_labels)
        print("Prediction accuracy on train data is: " + str(accuracy_train_data * 100) + "%")

        accuracy_test_data = perceptronClassifier.predict(test_data, test_labels)
        print("Prediction accuracy on test data is: " + str(accuracy_test_data * 100) + "%")

