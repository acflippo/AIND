"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np

DEBUG = False

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def manhattan_distance_score(game, this_player):
    """ Outputs a score equal to 6 - manhattan distance from the center of the
    board to the position of the player.

    The '8 - distance' normalizes the score so it reflects a
    lower score the further away the player is from the center.
    """

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(this_player)
    return float(abs(h - y) + abs(w - x))

def score_centrality(game, possible_moves):
    """ This function scores higher for center squares and decreases
    as it gets closer to the edge of the board.  Here are the score per square
    on the account on how many possible moves are available from that squares
    when the board is empty.

    2 | 3 | 4 | 4 | 4 | 3 | 2
    3 | 4 | 6 | 6 | 6 | 4 | 3
    4 | 6 | 8 | 8 | 8 | 6 | 4
    4 | 6 | 8 | 8 | 8 | 6 | 4
    4 | 6 | 8 | 8 | 8 | 6 | 4
    3 | 4 | 6 | 6 | 6 | 4 | 3
    2 | 3 | 4 | 4 | 4 | 3 | 2

    This function took a lot of time, now I rest.
    (\(\  zzz
    (-.-)
    o_(")(")

    """

    # Generalize to work with board of any size greater than 5x5
    w, h = game.width, game.height
    mid_h = int(np.floor(h/2.0))
    mid_w = int(np.floor(w/2.0))

    # Find a list of mid-point cells excluding first 2 cells in position 0,1
    # excluding last 2 positions width and width-1 or height and height-1
    mid_h_elem = list(range(2, h-2))
    mid_w_elem = list(range(2, w-2))

    score = 0
    for move in possible_moves:
        if move in [(0,0),(0,w-1),(w-1,0),(w-1,w-1)]:
            score += 2
        elif move in [(0,1),(0,w-2),(1,0),(1,w-1),(h-2,0),(h-2,w-1),(h-1,1),(h-1,w-2)]:
            score += 3
        elif move[0] in mid_h_elem and move[1] in (0, w-1):
            score += 4
        elif move[1] in mid_w_elem and move[0] in (0, h-1):
            score += 4
        elif move[0] in mid_h_elem and move[1] in (0, w-2):
            score += 6
        elif move[1] in mid_w_elem and move[0] in (0, h-2):
            score += 6
        else:
            score += 8

    return float(score)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_score = manhattan_distance_score(game, player)
    opp_score = manhattan_distance_score(game, game.get_opponent(player))

    return float(opp_score - my_score)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_score = len(game.get_legal_moves(player)) - manhattan_distance_score(game, player)
    opp_score = len(game.get_legal_moves(game.get_opponent(player))) - manhattan_distance_score(game, game.get_opponent(player))

    return float(my_score - opp_score)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_legal_moves = game.get_legal_moves(player)
    opp_legal_moves = game.get_legal_moves(game.get_opponent(player))

    my_score = score_centrality(game, my_legal_moves)
    opp_score = score_centrality(game, opp_legal_moves)

    return my_score - opp_score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    # MinimaxPlayer class' get_move
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is expire.
            self.best_move = self.minimax(game, self.search_depth)
            return self.best_move

        except SearchTimeout:
            pass

        return best_move

    # MinimaxPlayer class' minimax
    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Check if there are any legal moves available
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)

        # initialize best_score and its associated move for comparison
        max_score = float("-inf")
        self.best_move = legal_moves[0]

        for m in legal_moves:
            self.check_time()
            this_score = self.min_value(game.forecast_move(m), depth-1)
            if this_score > max_score:
                max_score = this_score
                self.best_move = m

        return self.best_move


    # MinimaxPlayer class' check_time
    def check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    # MinimaxPlayer class' min_value
    def min_value(self, game, depth):

        if depth == 0:
            return self.score(game, self)

        min_score = float("inf")
        for m in game.get_legal_moves():
            self.check_time()
            min_score = min(min_score, self.max_value(game.forecast_move(m), depth-1))

        return min_score

    # MinimaxPlayer class' max_value
    def max_value(self, game, depth):

        if depth == 0:
            return self.score(game, self)

        max_score = float("-inf")
        for m in game.get_legal_moves():
            self.check_time()
            max_score = max(max_score, self.min_value(game.forecast_move(m), depth-1))

        return max_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    # AlphaBetaPlayer Class' get_move
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)
        else:
            best_move = legal_moves[0]

        try:
            depth = 0
            while True:
                self.check_time()
                best_move = self.alphabeta(game, depth + 1)
                if DEBUG:
                    print("get_move: self.alphabeta( )")
                depth += 1

        except SearchTimeout:
            pass

        return best_move

    # AlphaBetaPlayer Class' alphabeta
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Check if there are any legal moves available
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)
        else:
            self.best_move = legal_moves[0]

        for m in legal_moves:
            this_score = self.min_value(game.forecast_move(m), depth-1, alpha, beta)
            if DEBUG:
                print("alphabeta: self.min_value( )")
            if this_score > alpha:
                alpha = this_score
                self.best_move = m

        return self.best_move

    # AlphaBetaPlayer Class' min_value
    def min_value(self, game, depth, alpha, beta):

        self.check_time()

        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)

        for m in game.get_legal_moves():
            self.check_time()
            this_score = self.max_value(game.forecast_move(m), depth-1, alpha, beta)

            if this_score < beta:
                beta = this_score
                # Prune and done
                if beta <= alpha:
                    if DEBUG:
                        print("alphabeta: min_value: pruning")
                    return float(beta)

        return float(beta)

    # AlphaBetaPlayer Class' max_value
    def max_value(self, game, depth, alpha, beta):

        self.check_time()

        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)

        for m in game.get_legal_moves():
            self.check_time()
            this_score = self.min_value(game.forecast_move(m), depth-1, alpha, beta)

            if this_score > alpha:
                alpha = this_score
                # Prune and done
                if alpha >= beta:
                    if DEBUG:
                        print("alphabeta: max_value: pruning")
                    return float(alpha)

        return float(alpha)

        # AlphaBetaPlayer Class' check_time
    def check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    # AlphaBetaPlayer Class' terminal_test
    def terminal_test(self, game):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        return not bool(game.get_legal_moves())
