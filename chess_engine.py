import chess

class GameState:
    trans = {'b': 'Black', 'w': "White"}

    def __init__(self):
        self.board = chess.Board()
        self.turn = 'w'
        self.moveLog = []

    def getValidMoves(self):
        return list(self.board.legal_moves)

    def makeMove(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
            self.moveLog.append(move)

    def undoMove(self):
        if self.board.move_stack:
            move = self.board.pop()
            self.moveLog.pop()

    def getMoveNotation(self):
        s = '{0:4}{1:7}{2:7}'.format("", "White", "Black")
        move_turn = 0
        for move in self.moveLog:
            if move_turn % 2 == 0:
                turn = f'{str(move_turn // 2 + 1)}.'
                s += '\n{0:4}'.format(turn)
            s += '{0:7}'.format(move.uci())
            move_turn += 1
        return s

    def getTurn(self):
        s = f"Turn: {self.trans[self.turn]}"
        return s

    def getFen(self):
        return self.board.fen()

    def getMoveFromUci(self, uci_move):
        try:
            move = chess.Move.from_uci(uci_move)
            if move in self.board.legal_moves:
                return move
            return None
        except ValueError:
            return None

    def uciToPolicyIndex(self, uci_move):
        legal_moves = list(self.board.legal_moves)
        for idx, move in enumerate(legal_moves):
            if move.uci() == uci_move:
                return idx
        return None

    def isCheckmate(self):
        return self.board.is_checkmate()

    def isStalemate(self):
        return self.board.is_stalemate()

    def isInsufficientMaterial(self):
        return self.board.is_insufficient_material()

    def isGameOver(self):
        return self.board.is_game_over()

    def getPieces(self):
        return self.board.piece_map()

    def getPieceAt(self, square):
        return self.board.piece_at(square)

    def isInCheck(self):
        return self.board.is_check()

    def getScore(self):
        value_map = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0
        }
        score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                symbol = piece.symbol().lower()
                value = value_map.get(symbol, 0)
                score += value if piece.color == chess.WHITE else -value
        return score

    def __str__(self):
        return str(self.board)

    def __repr__(self):
        return self.__str__()

class Move:
    """
    Class represent a move in game.
    It contains:  starting point coordinates, ending point coordinates, piece move and piece is captured
    """
    _rankMap = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
    _fileMap = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

    def __init__(self, sqStart, sqEnd, board, enPassantSquare=(), is_castle_move=False):
        self.sqStart = sqStart
        self.sqEnd = sqEnd
        self.movePiece = board[sqStart[0]][sqStart[1]]
        self.capturedPiece = board[sqEnd[0]][sqEnd[1]]

        self.isPawnPromotion = self.movePiece[1] == 'p' and self.sqEnd[0] in (0, 7)

        self.isEnpassant = (enPassantSquare != ())
        if self.isEnpassant:
            rival = {'w': 'b', 'b': 'w'}
            self.capturedPiece = f'{rival[self.movePiece[0]]}p'

        self.is_castle_move = is_castle_move

        self.moveID = 1000 * self.sqStart[0] + 100 * self.sqStart[1] + 10 * self.sqEnd[0] + self.sqEnd[1]

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    def getRankFile(self, r, c):
        return self._fileMap[c] + str(self._rankMap[r])

    def getChessNotation(self):

        if self.isPawnPromotion:
            return self.getRankFile(self.sqEnd[0], self.sqEnd[1]) + "Q"

        if self.is_castle_move:
            if self.sqEnd[1] == 1:
                return "0-0-0"
            else:
                return "0-0"

        if self.isEnpassant:
            return self.getRankFile(self.sqStart[0], self.sqStart[1])[0] + "x" + self.getRankFile(self.sqEnd[0],
                                                                                                  self.sqEnd[
                                                                                                      1]) + " e.p."
        if self.capturedPiece != "--":
            if self.movePiece[1] == "p":
                return self.getRankFile(self.sqStart[0], self.sqStart[1])[0] + "x" + self.getRankFile(self.sqEnd[0],
                                                                                                      self.sqEnd[1])
            else:
                return self.movePiece[1] + "x" + self.getRankFile(self.sqEnd[0], self.sqEnd[1])
        else:
            if self.movePiece[1] == "p":
                return self.getRankFile(self.sqEnd[0], self.sqEnd[1])
            else:
                return self.movePiece[1] + self.getRankFile(self.sqEnd[0], self.sqEnd[1])


class CastleRights:
    def __init__(self, wks, bks, wqs, bqs):
        self.wks = wks
        self.bks = bks
        self.wqs = wqs
        self.bqs = bqs

    def __eq__(self, other):
        if self.wks == other.wks and self.bks == other.bks and self.wqs == other.wqs and self.bqs == other.bqs:
            return True
        return False
