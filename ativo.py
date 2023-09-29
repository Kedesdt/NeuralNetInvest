

class Ativo():

    def __init__(self, valor, nome):

        self.valor_inicial = valor
        self.nome = nome
        self.valor = valor
        self.cotas_compradas = 0
        self.predicao = 0
        self.comprado = False
        self.tempo = 0