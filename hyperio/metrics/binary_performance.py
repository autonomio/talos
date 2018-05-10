class BinaryPerformance():

    def __init__(self, y_pred, y_val):

        self.y_pred = y_pred
        self.y_val = y_val

        self.trues_and_falses()
        self.f1score()
        self.balance()
        self.one_rule()
        self.zero_rule()

    def f1score(self):

        '''Computes fscore when possible'''

        if sum(self.y_pred) == len(self.y_pred):
            if sum(self.y_val) != len(self.y_val):
                self.result = '_all_ones_'
                return

        try:
            self.precision = self.tp / (self.tp + self.fp)
            1 / self.precision
        except ZeroDivisionError:
            self.result = '_all_zeros_'
            return

        try:
            self.recall = self.tp / (self.tp + self.fn)
            1 / self.recall
        except ZeroDivisionError:
            self.result = '_no_ones_'
            return

        try:
            f1 = 2 * ((self.precision * self.recall) / (self.precision + self.recall))
            self.result = f1
        except ZeroDivisionError:
            return


    def trues_and_falses(self):

        '''Returns tp, tn, fp, and fn values'''

        self.pos = sum(self.y_val)
        self.neg = len(self.y_val) - sum(self.y_val)

        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

        # then we iterate through the predictions
        for i in range(len(self.y_val)):

            if self.y_pred[i] == 1 and self.y_val[i] == 1:
                self.tp += 1
            elif self.y_pred[i] == 1 and self.y_val[i] == 0:
                self.fn += 1
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
