import math
from collections import defaultdict
import numpy as np


def file_reader(file_path, label):
    list_of_lines = []
    list_of_labels = []

    for line in open(file_path):
        line = line.strip()
        if line=="":
            continue
        list_of_lines.append(line)
        list_of_labels.append(label)

    return (list_of_lines, list_of_labels)


def data_reader(source_directory):
    positive_file = source_directory+"Positive.txt"
    (positive_list_of_lines, positive_list_of_labels)=file_reader(file_path=positive_file, label=1)

    negative_file = source_directory+"Negative.txt"
    (negative_list_of_lines, negative_list_of_labels)=file_reader(file_path=negative_file, label=-1)

    neutral_file = source_directory+"Neutral.txt"
    (neutral_list_of_lines, neutral_list_of_labels)=file_reader(file_path=neutral_file, label=0)

    list_of_all_lines = positive_list_of_lines + negative_list_of_lines + neutral_list_of_lines
    list_of_all_labels = np.array(positive_list_of_labels + negative_list_of_labels + neutral_list_of_labels)

    return list_of_all_lines, list_of_all_labels


def evaluate_predictions(test_set,test_labels,trained_classifier):
    correct_predictions = 0
    predictions_list = []
    prediction = -1
    for dataset,label in zip(test_set, test_labels):
        probabilities = trained_classifier.predict(dataset)
        if probabilities[0] >= probabilities[1] and probabilities[0] >= probabilities[-1]:
            prediction = 0
        elif  probabilities[1] >= probabilities[0] and probabilities[1] >= probabilities[-1]:
            prediction = 1
        else:
            prediction=-1
        if prediction == label:
            correct_predictions += 1
            predictions_list.append("+")
        else:
            predictions_list.append("-")
    
    print("Total Sentences correctly: ", len(test_labels))
    print("Predicted correctly: ", correct_predictions)
    print("Accuracy: {}%".format(round(correct_predictions/len(test_labels)*100,5)))

    return predictions_list, round(correct_predictions/len(test_labels)*100)


class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = []
        self.conditional = []
        self.V = []
        self.n = n_gram

    def word_tokenization_dataset(self, training_sentences):
        training_set = list()
        for sentence in training_sentences:
            cur_sentence = list()
            for word in sentence.split(" "):
                cur_sentence.append(word.lower())
            training_set.append(cur_sentence)
        return training_set

    def word_tokenization_sentence(self, test_sentence):
        cur_sentence = list()
        for word in test_sentence.split(" "):
            cur_sentence.append(word.lower())
        return cur_sentence

    def compute_vocabulary(self, training_set):
        vocabulary = set()
        for sentence in training_set:
            for word in sentence:
                vocabulary.add(word)
        V_dictionary = dict()
        dict_count = 0
        for word in vocabulary:
            V_dictionary[word] = int(dict_count)
            dict_count += 1
        return V_dictionary

    def train(self, training_sentences, training_labels):
        
        # See the HW_3_How_To.pptx for details
        
        # Get number of sentences in the training set
        N_sentences = len(training_sentences)

        # This will turn the training_sentences into the format described in the HW_3_How_To.pptx
        training_set = self.word_tokenization_dataset(training_sentences)

        # Get vocabulary (dictionary) used in training set
        self.V = self.compute_vocabulary(training_set)

        # Get set of all classes
        all_classes = set(training_labels)

        #-----------------------#
        #-------- TO DO (begin) --------#
        # Note that, you have to further change each sentence in training_set into a binary BOW representation, given self.V

        matrix=np.empty([len(training_set), len(self.V)])
        for i in range(len(training_set)):
            for j in range(len(self.V)):
                matrix[i][j]=0;

        for i in range(len(training_set)):
            for m in range(len(training_set[i])):
                index=self.V.get(training_set[i][m],-1)
                if(index!=-1):
                    matrix[i][index]=1
        # Compute the conditional probabilities and priors from training data, and save them in:
        # self.prior
        # self.conditional

        negative=0
        neutral=0
        positive=0

        for i in range(len(training_labels)):
            if(training_labels[i]==0):
                neutral=neutral+1
        for i in range(len(training_labels)):
            if(training_labels[i]==1):
                positive=positive+1
        for i in range(len(training_labels)):
            if(training_labels[i]==-1):
                negative=negative+1
        self.prior=[positive/len(training_labels),neutral/len(training_labels),negative/len(training_labels)]

        self.conditional=np.empty([3, len(self.V)])

        neutralcon=[0 for i in range(len(self.V))]
        positivecon=[0 for i in range(len(self.V))]
        negativecon=[0 for i in range(len(self.V))]
        alpha=.25;
        for i in range(len(training_labels)):
            if(training_labels[i]==0):
                for j in range(len(matrix[i])):
                    if(matrix[i][j]==1):
                        neutralcon[j]=(neutralcon[j]+1)
            if(training_labels[i]==1):
                for j in range(len(matrix[i])):
                    if(matrix[i][j]==1):
                        positivecon[j]=(positivecon[j]+1)
            if(training_labels[i]==-1):
                for j in range(len(matrix[i])):
                    if(matrix[i][j]==1):
                        negativecon[j]=(negativecon[j]+1)
        for j in range(len(matrix[i])):
            negativecon[j]=(negativecon[j]+alpha*negative)/(negative+20*alpha*negative)
            positivecon[j]=(positivecon[j]+alpha*positive)/(positive+20*alpha*positive)
            neutralcon[j]=(neutralcon[j]+alpha*neutral)/(neutral+20*alpha*neutral)

        self.conditional=[positivecon,neutralcon,negativecon]
        # You can use any data structure you want.
        # You don't have to return anything. self.conditional and self.prior will be called in def predict():



        # -------- TO DO (end) --------#
        

    def predict(self, test_sentence):

        # The input is one test sentence. See the HW_3_How_To.pptx for details
        
        # Your are going to save the log probability for each class of the test sentence. See the HW_3_How_To.pptx for details

        # This will tokenize the test_sentence: test_sentence[n] will be the "n-th" word in a sentence (n starts from 0)
        test_sentence = self.word_tokenization_sentence(test_sentence)

        #-----------------------#
        #-------- TO DO (begin) --------#
        # Based on the test_sentence, please first turn it into the binary BOW representation (given self.V) and compute the log probability
        # Please then use self.prior and self.conditional to calculate the log probability for each class. See the HW_3_How_To.pptx for details 
        matrix=np.empty([len(self.V)])
        for j in range(len(self.V)):
            matrix[j]=0;
        for m in range(len(test_sentence)):
                index=self.V.get(test_sentence[m],-1)
                if(index!=-1):
                    matrix[index]=1

        poslog=np.log(self.prior[0])
        neutlog=np.log(self.prior[1])
        neglog=np.log(self.prior[2])

        for i in range(len(self.V)):
            if(matrix[i]!=0):
                poslog=poslog+np.log(self.conditional[0][i])
                neutlog=neutlog+np.log(self.conditional[1][i])
                neglog = neglog + np.log(self.conditional[2][i])
            else:
                poslog = poslog + np.log(1-self.conditional[0][i])
                neutlog = neutlog + np.log(1-self.conditional[1][i])
                neglog = neglog + np.log(1-self.conditional[2][i])
        # Return a dictionary of log probability for each class for a given test sentence:
        # e.g., {0: -39.39854137691295, 1: -41.07638511893377, -1: -42.93948478571315}
        # Please follow the PPT to first perform log (you may use np.log) to each probability term and sum them.
        label_probability = {1: poslog, 0: neutlog, -1: neglog}





        # -------- TO DO (end) --------#

        return label_probability


if __name__ == '__main__':
    train_folder = "data-sentiment/train/"
    test_folder = "data-sentiment/test/"

    training_sentences, training_labels = data_reader(train_folder)
    test_sentences, test_labels = data_reader(test_folder)

    NBclassifier = NaiveBayesClassifier(n_gram=1)
    NBclassifier.train(training_sentences,training_labels)

    results, acc = evaluate_predictions(test_sentences, test_labels, NBclassifier)

