import pandas
import nn
import random
import yfinance as yf
import pandas as pd
import investpy as inv
import numpy


IBOV_JAN2022 = 104700
IBOV_JAN2023 = 109900
QUANTIDADE = 10
QUANTIDADE_DE_ACOES = 5

CAPITAL_INICIAL = 100000

compras = 0
vendas = 0
patrimonio = CAPITAL_INICIAL

class Ativo():

    def __init__(self, valor, nome):

        self.nome = nome
        self.valor = valor
        self.cotas_compradas = 0
        self.predicao = 0


class Investidor:

    def __init__(self, rede, df):

        self.rede = rede
        self.df = df

        self.ativos = []

        for i in self.df.keys():
            self.ativos.append(Ativo(0, i))
        self.nomes_ativos = [ativo.nome for ativo in self.ativos]
        self.patrimonio = CAPITAL_INICIAL
        self.dinheiro_disponivel = self.patrimonio
        self.dinheiro_investido = 0
        self.compras = 0
        self.vendas = 0
        self.valor_comprado = 0
        self.valor_vendido = 0

    def get_by_name(self, name):

        for i in self.ativos:
            if i.nome == name:
                return i

    def get_by_pre(self, pre):

        for i in self.ativos:
            if i.predicao == pre:
                return i

    def vende_ativo(self, ativo):

        valor = ativo.cotas_compradas * ativo.valor
        self.dinheiro_disponivel += valor
        ativo.cotas_compradas = 0
        if valor > 0:
            self.vendas += 1
        self.valor_vendido += valor

    def compra_ativo(self, ativo, valor):

        self.compras += 1
        self.valor_comprado += valor
        self.dinheiro_disponivel -= valor
        ativo.cotas_compradas += valor / ativo.valor


    def invest(self):

        for i in range(29, len(self.df) - 1):

            d = self.df[i - 29:i + 1].copy()

            saidas = {}
            predicoes = []

            for key in self.nomes_ativos:

                if key == "Date":
                    continue

                ativo = self.get_by_name(key)
                Date = d.axes[0][-1]
                ativo.valor = d[key].values[-1]
                entrada = d[key].values / max(d[key].values)
                entrada = entrada.reshape(len(entrada), 1)
                saidas[key] = neuralnet.feedforward(entrada)
                ativo.valor = d[key].values[-1]
                ativo.predicao = saidas[key][0][0]
                predicoes.append(saidas[key][0][0])

            predicoes.sort()

            ativos_na_ordem = []

            for i in predicoes:
                ativos_na_ordem.append(self.get_by_pre(i))

            for i in range(len(ativos_na_ordem)):
                if i < QUANTIDADE_DE_ACOES:
                    self.vende_ativo(ativos_na_ordem[i])

            q = QUANTIDADE - QUANTIDADE_DE_ACOES
            valor = self.dinheiro_disponivel / q
            if valor > 0:
                for i in range(len(ativos_na_ordem)):
                    if i >= QUANTIDADE_DE_ACOES:
                        self.compra_ativo(ativos_na_ordem[i], valor)

            self.dinheiro_investido = 0
            for ativo in self.ativos:
                self.dinheiro_investido += ativo.cotas_compradas * ativo.valor
            self.patrimonio = self.dinheiro_disponivel + self.dinheiro_investido

        self.print()

    def print(self):

        print("Patrimonio Final: ", self.patrimonio,
              "\nPatrimonio Inicial: ", CAPITAL_INICIAL,
              "\nGanho: ", self.patrimonio / CAPITAL_INICIAL,
              "\nRelação RedeNeural/Ibovespa: ",
              (self.patrimonio / CAPITAL_INICIAL) / (IBOV_JAN2023 / IBOV_JAN2022),
              "\nCompras: ", self.compras,
              "\nVendas: ", self.vendas,
              "\nValor_Comprado: ", self.valor_comprado,
              "\nValor_Vendido: ", self.valor_vendido)




br = inv.stocks.get_stocks(country='brazil')
carteira = []

i = 0
for a in br['symbol']:
    if len(a) <= 5:
        carteira.append(a+".SA")
        i += 1

    if i >= QUANTIDADE:
        pass

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


dt = yf.download(carteira, start='2012-01-01', end="2022-01-01")["Adj Close"]

data = []
for key in dt.keys():

    if key == "Date":
        continue
    d = dt[key].copy()
    #print(d)
    d = d.dropna(axis=0, how='all')
    #print(d)
    for i in range(29, len(d) - 4):
        entrada = d[i-29:i+1].values / max(d[i-29:i+1].values)
        entrada = entrada.reshape(len(entrada), 1)
        saida = []
        ranges = [1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1]
        for j in range(4):
            saida.append([1 if d[i+1+j] > d[i] else 0])

        saida = numpy.array(saida)
        data.append([entrada, saida])

    print(key, " Feito.")

random.shuffle(data)

indice = int(len(data) * 0.8)

td = data[indice:]
d = data[:indice]

neuralnet = nn.Network([30, 25, 20, 4])
neuralnet.SGD(d, 200, 500, 0.005, td)


df = yf.download(carteira, start='2022-01-01', end="2023-01-01")["Adj Close"]

df = df.dropna(axis=0, how='all')


inv = Investidor(neuralnet, df)
inv.invest()








