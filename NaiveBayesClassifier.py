from collections import defaultdict
import math


class NaiveBayesClassifier():
    prior = None
    counter = None
    def train(self, train_data, train_labels, smooth_alpha):
        prior = defaultdict(int)
        counter = {i: {x: 0 for x in range(10)} for i in range(784)}
        for data, label in zip(train_data, train_labels):
            prior[label] += 1
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] != ' ':
                        counter[(len(data[i]) * i) +  j][label] += 1

        prior = self.normalize_dict(prior)
        for i in range(len(data)):
            for j in range(len(data[i])):
                for label in range(10):
                    counter[(len(data[i]) * i) +  j][label] = self.calcProbability(counter[(28 * i) +  j][label], len(train_data), 2, smooth_alpha)

        # Save prior and counter in object
        self.prior = prior
        self.counter = counter


    def predict(self, test_data, test_labels):
        correct_precitions = 0
        for data, label in zip(test_data, test_labels):
            probs = [math.log(self.prior[x]) for x in range(10)]

            for i in range(len(data)):
                for j in range(len(data[i])):
                    for k in range(10):
                        prob = self.counter[(len(data[i]) * i) +  j][k] if data[i][j] != ' ' else \
                            1 - self.counter[(len(data[i]) * i) +  j][k]

                        probs[k] =  probs[k] + math.log(prob)
                    
            predicted_label = probs.index(max(probs))

            if predicted_label == label:
                correct_precitions += 1
            
        return correct_precitions / len(test_data)

    def normalize_dict(self, d):
        total = sum(d.values())
        for each in d:
            d[each] /= total
        return d


    def calcProbability(self, occurency, total_samples, class_numbers, smooth_alpha):
        # Return the probability with k-smooth 
        return (occurency + smooth_alpha) / (total_samples + (class_numbers * smooth_alpha))

