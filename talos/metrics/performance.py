from keras.utils import to_categorical


class Performance():

    '''A modified fscore where the predictions can be
    single-label, multi-label, or multi-category, and
    each will be treated in a way that allows comparison
    (apples-to-apples) between different prediction tasks.'''

    def __init__(self, y_pred, y_val, shape, y_max):

        self.y_pred = y_pred
        self.y_val = y_val
        self.shape = shape

        self.classes = y_max + 1

        if self.shape == 'binary_class':
            self.binary_class()
        elif self.shape == 'multi_class':
            self.multi_class()
        elif self.shape == 'multi_label':
            self.multi_label()

        self.trues_and_falses()
        self.f1score()
        self.balance()
        self.one_rule()
        self.zero_rule()

    def multi_class(self):

        '''For one-hot encoded'''

        self.y_pred = self.y_pred.flatten('F')
        self.y_val = self.y_val.flatten('F')

    def binary_class(self):

        '''For single column, single label'''

        return

    def multi_label(self):

        '''For many labels in a single column'''

        self.y_pred = to_categorical(self.y_pred, num_classes=self.classes)
        self.y_val = to_categorical(self.y_val, num_classes=self.classes)

        self.multi_class()

    def f1score(self):

        '''Computes fscore when possible'''

        if sum(self.y_pred) == len(self.y_pred):
            if sum(self.y_val) != len(self.y_val):
                self.result = '_warning_all_ones_'
                return
        elif sum(self.y_pred) == 0:
            if sum(self.y_val) != 0:
                self.result = '_warning_all_zeros_'
            elif sum(self.y_val) == 0:
                self.result = 1
            return
        try:
            self.precision = self.tp / (self.tp + self.fp)
            1 / self.precision
        except ZeroDivisionError:
            self.result = '_warning_no_true_positive'
            return

        try:
            self.recall = self.tp / (self.tp + self.fn)
            1 / self.recall
        except ZeroDivisionError:
            self.result = '_warning_3'
            return

        try:
            f1 = 2 * ((self.precision * self.recall) /
                      (self.precision + self.recall))
            self.result = f1
        except ZeroDivisionError:
            return

    def trues_and_falses(self):

        '''Returns tp, tn, fp, and fn values'''

        self.pos = sum(self.y_val)
        self.neg = len(self.y_val) - sum(self.y_val)

        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

        # then we iterate through the predictions
        for i in range(len(self.y_pred)):
            if self.y_pred[i] == 1 and self.y_val[i] == 1:
                self.tp += 1
            elif self.y_pred[i] == 1 and self.y_val[i] == 0:
                self.fp += 1
            elif self.y_pred[i] == 0 and self.y_val[i] == 0:
                self.tn += 1
            elif self.y_pred[i] == 0 and self.y_val[i] == 1:
                self.fn += 1

    def balance(self):

        '''Counts the balance between 0s and 1s'''

        self.balance = self.pos / (self.pos + self.neg)

    def zero_rule(self):

        '''Returns accuracy for all 0s'''

        self.zero_rule = 1 - self.balance

    def one_rule(self):

        '''Returns accuracy for all 1s'''

        self.one_rule = self.balance
