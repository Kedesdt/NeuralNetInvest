import pandas
import nn
import random
import yfinance as yf
import pandas as pd
import investpy as inv
import numpy


IBOV_JAN2022 = 104700
IBOV_JAN2023 = 109900
QUANTIDADE = 20
QUANTIDADE_DE_ACOES = 10

DITR = '2021-01-02'
DFTR = '2022-01-02'

DITES = '2022-01-02'
DFTES = '2022-07-02'

EPOCAS = 30000
MINIBATCH = 30
TAXA = 0.001
DIAS = 1
NEURONIOSDEENTRADA = 60

CAPITAL_INICIAL = 100000

compras = 0
vendas = 0
patrimonio = CAPITAL_INICIAL

class Ativo():

    def __init__(self, valor, nome):

        self.valor_inicial = valor
        self.nome = nome
        self.valor = valor
        self.cotas_compradas = 0
        self.predicao = 0
        self.comprado = False
        self.tempo = 0


class Investidor:

    def __init__(self, rede, df, dv):

        self.rede = rede
        self.df = df
        self.dv = dv

        self.trades = []

        self.ativos = []

        for i in self.df.keys():
            self.ativos.append(Ativo(self.df[i].values[29], i))
        self.nomes_ativos = [ativo.nome for ativo in self.ativos]
        self.patrimonio = CAPITAL_INICIAL
        self.dinheiro_disponivel = self.patrimonio
        self.dinheiro_investido = 0
        self.compras = 0
        self.vendas = 0
        self.valor_comprado = 0
        self.valor_vendido = 0
        self.ganho_medio = None

    def get_by_name(self, name):

        for i in self.ativos:
            if i.nome == name:
                return i

    def get_by_pre(self, pre):

        for i in self.ativos:
            if i.predicao == pre:
                return i

    def vende_ativo(self, ativo, i):

        valor = ativo.cotas_compradas * ativo.valor
        self.dinheiro_disponivel += valor
        ativo.cotas_compradas = 0
        if valor > 0:
            self.vendas += 1
            if ativo.comprado:
                t = i - ativo.tempo
                self.trades.append(t)
                ativo.tempo = 0
                ativo.comprado = False
        self.valor_vendido += valor

    def compra_ativo(self, ativo, valor, i):

        ativo.tempo = i
        self.compras += 1
        self.valor_comprado += valor
        self.dinheiro_disponivel -= valor
        ativo.cotas_compradas += valor / ativo.valor
        ativo.comprado = True

    def calcula_medio(self):
        m = 0
        for i in self.ativos:
            m += i.valor / i.valor_inicial
        m /= len(self.ativos)
        return m

    def invest(self):

        for i in range(29, len(self.df) - 1):

            d = self.df[i - 29:i + 1].copy()
            dv = self.dv[i - 29:i + 1].copy()
            saidas = {}
            predicoes = []

            for key in self.nomes_ativos:

                if key == "Date":
                    continue

                ativo = self.get_by_name(key)
                Date = d.axes[0][-1]
                ativo.valor = d[key].values[-1]
                entrada = d[key].values < d[key].values[-1]
                entrada = numpy.append(entrada, d[key].values < d[key].values[-1])
                entrada = entrada.reshape(len(entrada), 1)
                saidas[key] = neuralnet.feedforward(entrada)
                ativo.valor = d[key].values[-1]
                ativo.predicao = saidas[key][0][0]
                predicoes.append(saidas[key][0][0])

            predicoes.sort()

            if 'nan' in predicoes:
                continue

            ativos_na_ordem = []
            comprar = []

            err = False
            for j in predicoes:
                ativos_na_ordem.append(self.get_by_pre(j))
                if ativos_na_ordem[-1] is None:
                    err = True
            if err:
                continue
            for j in range(len(ativos_na_ordem)):
                #if i < QUANTIDADE_DE_ACOES:
                #    self.vende_ativo(ativos_na_ordem[i])
                if ativos_na_ordem[j].predicao < 0.5:
                    self.vende_ativo(ativos_na_ordem[j], i)
                else:
                    comprar.append(ativos_na_ordem[j])

            #q = QUANTIDADE - QUANTIDADE_DE_ACOES
            q = len(comprar)
            if not q:
                q = 1
            valor = self.dinheiro_disponivel / q
            if valor > 0:
                for j in comprar:
                    #if i >= QUANTIDADE_DE_ACOES:
                    self.compra_ativo(j, valor, i)

            self.dinheiro_investido = 0
            for ativo in self.ativos:
                self.dinheiro_investido += ativo.cotas_compradas * ativo.valor
            self.patrimonio = self.dinheiro_disponivel + self.dinheiro_investido

        self.ganho_medio = self.calcula_medio()

        self.print()

    def print(self):

        media = len(self.trades) / (len(self.df) / 30)
        if len(self.trades):
            tempomedio = sum(self.trades) / len(self.trades)
        else:
            tempomedio = 0

        print("Patrimonio Final: ", self.patrimonio,
              "\nPatrimonio Inicial: ", CAPITAL_INICIAL,
              "\nGanho: ", self.patrimonio / CAPITAL_INICIAL,
              "\nGanho medio: ", self.ganho_medio,
              "\nRelação RedeNeural/Ibovespa: ",
              (self.patrimonio / CAPITAL_INICIAL) / (GANHO_IBOV),
              "\nRelação RedeNeural/Ganho medio: ",
              (self.patrimonio / CAPITAL_INICIAL) / (self.ganho_medio),
              "\nCompras: ", self.compras,
              "\nVendas: ", self.vendas,
              "\nTempo médio trade : ", tempomedio, "dias",
              "\nMédia de : ", media, "trades por mês",
              "\nValor_Comprado: ", self.valor_comprado,
              "\nValor_Vendido: ", self.valor_vendido)


br = inv.stocks.get_stocks(country='brazil')
carteira = []

redes = {}

i = 0
for a in br['symbol']:
    carteira.append(a+".SA")
    i += 1

    if i >= QUANTIDADE:
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
#            "C1TV34.SA", "MOSC34.SA", "BRKM5.SA", "GOGl34.SA", 'ACWI11.SA']

carteira = ['AAPL34.SA', "GOGl34.SA", "MSFT34.SA", "AMZO34.SA"]
#carteira = ['AAPL34.SA', "GOGl34.SA"]

#carteira = ['ITUB4.SA', "BBDCA4.SA"]


data = yf.download(carteira, start=DITR, end=DFTR)
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
    for i in range(29, len(d) - DIAS):
        #entrada = d[i-29:i+1].values / max(d[i-29:i+1].values)
        e = d[i - 29:i + 1].values
        entrada = e < e[-1]
        edv = dv[i-29:i+1].values
        entrada = numpy.append(entrada, edv < edv[-1])
        entrada = entrada.reshape(len(entrada), 1)
        saida = []
        ranges = [1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1]
        for j in range(DIAS):
            saida.append([1 if d[i+1+j] > d[i] else 0])
        saida = numpy.array(saida)
        data.append([entrada, saida])

    print(key, " Feito.")

random.shuffle(data)

indice = int(len(data) * 0.8)

td = data[indice:]
#d = data[:indice]
d = data[:indice]

neuralnet = nn.Network([NEURONIOSDEENTRADA, 60, 60, DIAS])
neuralnet.SGD(d, EPOCAS, MINIBATCH, TAXA, td)


data = yf.download(carteira, start=DITES, end=DFTES)
df = data['Adj Close']
df = df.dropna(axis=0, how='all')
dv = data['Volume']
dv = dv.dropna(axis=0, how='all')

ibov = yf.download("^BVSP", start=DITES, end=DFTES)['Adj Close']
GANHO_IBOV = ibov.values[-1]/ibov.values[0]
QUANTIDADE = len(df.keys())
QUANTIDADE_DE_ACOES = int(QUANTIDADE/2)


inv = Investidor(neuralnet, df, dv)
inv.invest()









