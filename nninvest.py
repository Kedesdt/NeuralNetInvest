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

    carteira = ["ITSA4.SA", "SUZB3.SA", 'PETR4.SA', 'JBSS3.SA', "RAIL3.SA",
                "C1TV34.SA", "MOSC34.SA", "BRKM5.SA", "GOGl34.SA", 'ACWI11.SA']

    #carteira = ['AAPL34.SA', "GOGl34.SA", "MSFT34.SA", "AMZO34.SA"]
    #carteira = ['AAPL34.SA', "GOGl34.SA"]

    #carteira = ['ITUB4.SA', "BBDCA4.SA"]


    data = yf.download(carteira, start=constantes.DITR, end=constantes.DFTR)
    dt = data["Adj Close"]
    v = data["Volume"]


    data = []
    for key in dt.keys():

        if key == "Date":
            continue
        d = dt[key].copy()
        dv = v[key].copy()
        #print(d)
        d = d.dropna(axis=0, how='all')
        dv = dv.dropna(axis=0, how='all')
        #print(d)
        for i in range(29, len(d) - constantes.DIAS): # pegar do dia 29 até o penultimo dia
            #entrada = d[i-29:i+1].values / max(d[i-29:i+1].values)
            e = d[i - 29:i + 1].values #
            edv = dv[i - 29:i + 1].values
            entrada = geraEntrada(e, edv)
            saida = []
            #ranges = [1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1]
            for j in range(constantes.DIAS):
                saida.append([1 if d[i+1+j] > d[i] else 0])
            saida = numpy.array(saida)
            data.append([entrada, saida])

        print(key, " Feito.")

    random.shuffle(data) # embaralha

    indice = int(len(data) * 0.8) # 80% para treinar a rede

    td = data[indice:] # data[80:] pego 20% para testar se a rede está aprendendo
    #d = data[:indice]
    d = data[:indice] # 80% daos dados para treinar a rede

    neuralnet = nn.Network([constantes.NEURONIOSDEENTRADA, 40, 20, constantes.DIAS])
    neuralnet.SGD(d, constantes.EPOCAS, constantes.MINIBATCH, constantes.TAXA, td)
    neuralnet.save()

    data = yf.download(carteira, start=constantes.DITES, end=constantes.DFTES)
    df = data['Adj Close']
    df = df.dropna(axis=0, how='all')
    dv = data['Volume']
    dv = dv.dropna(axis=0, how='all')

    ibov = yf.download("^BVSP", start=constantes.DITES, end=constantes.DFTES)['Adj Close']
    GANHO_IBOV = ibov.values[-1]/ibov.values[0]
    #constantes.QUANTIDADE = len(df.keys())
    #constantes.QUANTIDADE_DE_ACOES = int(constantes.QUANTIDADE/2)


    inv = Investidor(neuralnet, df, dv, GANHO_IBOV)
    inv.invest()

if __name__ == "__main__":
    main()









