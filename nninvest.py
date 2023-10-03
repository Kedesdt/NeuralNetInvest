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

    """    
    01 - Magazine Luiza (MGLU3) - Varejo
    02 - Americanas (AMER3) - Varejo
    03 - PetroRio (PRIO3) - Minas e gas
    04 - Eletrobras (ELET3)
    05 - Rede D'Or (RDOR3) - saúde
    06 - Embraer (EMBR3) - aeroespacial
    07 - Sabesp (SBSP3) - serviço
    08 - Totvs (TOTS3) - tecnologia
    09 - Vivo (VIVT3) - telefonia
    10 - Banco Santande (SANB3) - financeiro
    """

    carteira = ["MGLU3.SA", "AMER3.SA", 'PRIO3.SA', 'ELET3.SA', "RDOR3.SA",
                "EMBR3.SA", "SBSP3.SA", "TOTS3.SA", "VIVT3.SA", 'SANB3.SA']

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









