import pandas
import nn
import random
import yfinance as yf
import pandas as pd
import investpy as invest
import numpy
from investidor import Investidor
import constantes
from funcoes import *
import time
import config

class Ativo():

    def __init__(self, nome, data, rede):
        self.nome = nome
        self.data = data
        self.rede = rede
        self.td = None
        self.acertos = {}
        self.lucros = {}

def main():

    br = invest.stocks.get_stocks(country='brazil')
    carteira = []

    redes = {}

    i = 0
    for a in br['symbol']:
        carteira.append(a+".SA")
        i += 1

        if i >= constantes.QUANTIDADE:
            break

    print(carteira)

    """Itaúsa (ITSA4) – Financeiro: Brasil – 10%
    Suzano (SUZB3) – Papel e Celulose: Brasil – 10%
    Petrobras (PETR4) – Óleo e Gás: Brasil – 10%
    JBS (JBSS3) – Transportes: Brasil – 10%
    Rumo Logística (RAIL3) – Transportes: Brasil – 10%
    Goldman Sachs (C1TV34) – Financeiro: EUA – 10%
    The Mosaic (MOSC34) – Agro: EUA – 10%
    Braskem (BRKM5) – Hotéis: EUA – 10%
    Alphabet (GOGL34) – Tecnologia: EUA – 10%
    Trend ACWI (ACWI11) – Multisetorial: Mundo – 10%"""

    #carteira = ["ITSA4.SA", "SUZB3.SA", 'PETR4.SA', 'JBSS3.SA', "RAIL3.SA",
    #            "AAPL34.SA", "BRKM5.SA", "GOGl34.SA", 'ACWI11.SA', "ABEV3.SA",
    #            'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'B3SA3.SA', 'SANB11.SA',
    #            'BBAS3.SA', 'PCAR3.SA', 'GGBR4.SA', 'CSNA3.SA', 
    #            'CIEL3.SA', 'HYPE3.SA', 'ELET6.SA', 'UGPA3.SA']
    #carteira = ["ITSA4.SA", "SUZB3.SA", 'PETR4.SA', 'JBSS3.SA', "RAIL3.SA",
    #            "BRKM5.SA", "GOGl34.SA", "AAPL34.SA", "BBDCA4.SA", "ELET6.SA"]

    #carteira = ['AAPL34.SA', "GOGl34.SA", "MSFT34.SA", "AMZO34.SA"]
    #carteira = ['AAPL34.SA', "GOGl34.SA"]

    #carteira = ['ITUB4.SA', "BBDCA4.SA"]
    carteira = ['B3SA3.SA', "GOGl34.SA"]


    data = yf.download(carteira, start=constantes.DITR, end=constantes.DFTR)
    dt = data["Adj Close"]
    v = data["Volume"]

    dt = dt.dropna(axis=0, how='all')
    v = v.dropna(axis=0, how='all')
    ativos = []

    for key in dt.keys():
        data = []
        if key == "Date":
            continue
        d = dt[key].copy()
        dv = v[key].copy()
        #print(d)
        d = d.dropna()
        dv = dv.dropna()
        #print(d)
        for i in range(29, len(d) - constantes.DIAS): # pegar do dia 29 até o penultimo dia
            #entrada = d[i-29:i+1].values / max(d[i-29:i+1].values)
            e = d[i - 29:i + 1].values #
            edv = dv[i - 29:i + 1].values
            entrada = geraEntrada_relacao(e, edv)
            saida = []
            #ranges = [1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1]
            for j in range(constantes.DIAS):
                #saida.append([1 if d[i+1+j] > d[i] else 0])
                if d[i]:
                    saida.append([d[i + 1 + j] / d[i]])
                else:
                    saida.append([0])
            saida = numpy.array(saida)
            data.append([entrada, saida])

        ativo = Ativo(key, data, None)
        ativos.append(ativo)

        print(key, " Feito.")

    for ativo in ativos:
        #random.shuffle(ativo.data)  # embaralha

        indice = int(len(ativo.data) * 0.95)  # 80% para treinar a rede

        ativo.td = ativo.data[indice:]  # data[80:] pego 20% para testar se a rede está aprendendo
        # d = data[:indice]
        ativo.data = ativo.data[:indice]  # 80% daos dados para treinar a rede

    for j in range(constantes.NUMERODEREDES):
        for i in range(len(ativos)):
            ativos[i].acertos[j] = []
            ativos[i].lucros[j] = []

    data = yf.download(carteira, start=constantes.DITES, end=constantes.DFTES)
    df = data['Adj Close']
    df = df.dropna()
    dv = data['Volume']
    dv = dv.dropna()
    ibov = yf.download("^BVSP", start=constantes.DITES, end=constantes.DFTES)['Adj Close']
    ibov = ibov.dropna()
    GANHO_IBOV = ibov.values[-1] / ibov.values[29]

    for j in range(constantes.QUANTIDADEDEREPETICOES):

        for i in range(constantes.NUMERODEREDES):
            for ativo in ativos:
                config.inicial_config(str(i), ativo.nome)
                ativo.rede = nn.Network.load(str(i) + '/' + ativo.nome +"/nn.json")
                if ativo.rede:
                    print("Rede carregada")
                    print("Esta rede ja foi treinada %i vezes" %ativo.rede.epochs_trained)
                else:
                    print("Não existe Rede\nCriando nova")
                    ativo.rede = nn.Network([constantes.NEURONIOSDEENTRADA, 40, 20, constantes.DIAS])
                if j > 0:
                    print(ativo.nome)
                    ativo.rede.SGD(ativo.data, constantes.EPOCAS * j, constantes.MINIBATCH, constantes.TAXA, ativo.td)
                #nome = strstr(i) + ".json"
                ativo.rede.save(str(i) +"/" + ativo.nome +"/nn.json")


            # constantes.QUANTIDADE = len(df.keys())
            #constantes.QUANTIDADE_DE_ACOES = int(constantes.QUANTIDADE/2)

            inv = Investidor(df, dv, GANHO_IBOV, ibov, ID = i, ativos = ativos)
            inv.invest()

if __name__ == "__main__":
    
    main()