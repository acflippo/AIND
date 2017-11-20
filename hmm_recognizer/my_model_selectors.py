import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        # Model selection: The lower the AIC/BIC value the better the model (only
        # compare AIC with AIC and BIC with BIC values

        # To find a lowest possible BIC, set default to an extremely high number
        min_bic_score = np.inf
        best_model = None

        try:
            for n_comp in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n_comp)

                logL = model.score(self.X, self.lengths)
                n_features = len(self.lengths)
                logN = np.log(n_features)

                # n_params = number_of_hmm_parameters(n_comp, n_features)
                n_params = n_comp**2 + 2 * n_comp * n_features -1
                bic_score = -2 * logL + n_params * logN

                #print ('bic_score: ', bic_score)

                if bic_score < min_bic_score:
                    min_bic_score = bic_score
                    best_model = model
        except:
            pass

        return best_model

    # def number_of_hmm_parameters(self, n_components, n_features):
    #     # Per discussion https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/15
    #     return n_components**2 + 2 * n_components * n_features - 1


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf [1]
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf [2]
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    Note:
    The first term is a difference between the likelihood of the data, and the average of
    anti-likelihood terms where the anti-likelihood of the data X's except the Xi itself.
    (discussed on 3.2 Discriminative Information Criterion by Alain Biem from [2] above.)
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        # Find maximum likelihood, set the default score to an extremely small number
        max_dic_score = -1*np.inf
        best_model = None

        try:
            for n_comp in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n_comp)
                anti_likelihood_list = []

                # first term in DIC
                logL = model.score(self.X, self.lengths)

                for word in self.hwords:
                    # Find the logLikelihood for all the words in list except
                    # the word in the model in question
                    if word != self.this_word:
                        Xdata, xlengths = self.hwords[word]
                        anti_likelihood = model.score(Xdata, xlengths)
                        anti_likelihood_list.append(anti_likelihood)

                # second term in DIC
                avg_logL = np.mean(anti_likelihood_list)
                dic_score = logL - avg_logL

                if dic_score > max_dic_score:
                    max_dic_score = dic_score
                    best_model = model

        except:
            pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        # Find maximum likelihood, set the default score to an extremely small number
        max_cv_score = -1*np.inf
        best_model = None
        num_splits = 3

        try:
            for n_comp in range(self.min_n_components, self.max_n_components + 1):

                if len(self.sequences) > 2:
                    cv_score_list = []
                    # print("n_comp: ", n_comp, "len(self.sequences): ", len(self.sequences))

                    k_fold = KFold(n_splits = num_splits)

                    for train_idx, test_idx in k_fold.split(self.sequences):

                        self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                        X_test, len_test  = combine_sequences(test_idx, self.sequences)

                        # model = GaussianHMM(n_components=n_comp, covariance_type="diag", n_iter=1000,
                        #                         random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                        model = self.base_model(n_comp)
                        logL = model.score(X_test, len_test)
                        cv_score_list.append(logL)
                else:
                    return self.base_model(self.n_constant)

                if len(cv_score_list) > 0:
                    avg_cv_score = np.mean(cv_score_list)

                    if avg_cv_score > max_cv_score:
                        max_cv_score = avg_cv_score
                        best_model = model


        except:
            pass

        return best_model
