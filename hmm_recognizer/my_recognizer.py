import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for test_word, (X, lengths) in test_set.get_all_Xlengths().items():

        max_score = float('-inf')
        best_guess = None
        word_logL = {}

        for train_word, model in models.items():
            try:
                logL = model.score(X, lengths)
                word_logL[train_word] = logL
            except:
                word_logL[train_word] = float('-inf')

            if logL > max_score:
                max_score = logL
                best_guess = train_word

        probabilities.append(word_logL)
        guesses.append(best_guess)

    return probabilities, guesses
