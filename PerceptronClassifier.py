class PerceptronClassifier():
    weights = None

    def train(self, train_data, train_labels, epochs_number):
        self.weights = [[0 for i in range(784)] for j in range(10)]
        
        for epoch in range(epochs_number):
            for data, label in zip(train_data, train_labels):
                
                label_scores =  [0] * 10

                for i in range(len(data)):
                    for j in range(len(data[i])):
                        feature = 1 if data[i][j] != ' ' else 0
                        for k in range(10):
                            label_scores[k] += feature * self.weights[k][(28 * i) +  j]
                
                predicted_label = label_scores.index(max(label_scores))
                if predicted_label != label:
                    # Update weight for the wrong predicted label
                    for i in range(len(data)):
                        for j in range(len(data[i])):
                            feature = 1 if data[i][j] != ' ' else 0

                            self.weights[predicted_label][(28 * i) +  j] -= feature
                            self.weights[label][(28 * i) +  j] += feature


    def predict(self, test_data, test_labels):
        correct_precitions = 0
        for data, label in zip(test_data, test_labels):
            label_scores = [0 for x in range(10)]
            for i in range(len(data)):
                for j in range(len(data[i])):            
                    for k in range(10):
                        feature = 1 if data[i][j] != ' ' else 0
                        label_scores[k] += (feature * self.weights[k][(28 * i) +  j])

            predicted_label = label_scores.index(max(label_scores))
            if predicted_label == label:               
                correct_precitions += 1
            
        return correct_precitions / len(test_data)
            

