import random

class Ativo():

    def __init__(self, valor, nome, rede = None, data = None):

        self.valor_inicial = None
        self.cotas_compradas_se_media = None
        self.nome = nome
        self.valor = valor
        self.cotas_compradas = 0
        self.predicao = 0
        self.comprado = False
        self.tempo = 0
        self.rede = rede
        self.data = data
        
        random.shuffle(self.data)
        
        if self.data is not None:
            self.index = int(len(self.data) * 0.8)

        self.td = self.data[self.index:]
        self.data = self.data[:self.index]
        
