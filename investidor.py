from ativo import Ativo
from constantes import *
from funcoes import *
import pandas as pd
import matplotlib.pyplot as plt
import time

class Investidor:

    def __init__(self, rede, df, dv, ganho_ibov = 1, df_ibov = None, ID = 1):

        self.id = ID
        self.rede = rede
        self.df = df
        self.dv = dv
        self.df_ibov = df_ibov
        self.ganho_ibov = ganho_ibov
        self.trades = []
        self.dataframe = pd.DataFrame()
        self.dataframe_medio = pd.DataFrame()
        self.lista = []
        self.lista_ibov = []
        self.lista_medio = []
        self.nome = "Adj Close"
        self.datas = []
        self.ativos = []
        #self.patrimonio = CAPITAL_INICIAL
        self.patrimonio = self.df_ibov.values[29]
        self.patrimonio_medio = self.patrimonio
        self.dinheiro_disponivel = self.patrimonio
        self.dinheiro_investido = 0
        self.compras = 0
        self.vendas = 0
        self.valor_comprado = 0
        self.valor_vendido = 0
        self.ganho_medio = None

        for i in self.df.keys():
            self.ativos.append(Ativo(self.df[i].values[29], i))
        self.nomes_ativos = [ativo.nome for ativo in self.ativos]
        for ativo in self.ativos:
            ativo.cotas_compradas_se_medio = (self.patrimonio / len(self.ativos)) / ativo.valor

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
                date = d.axes[0][-1]
                ativo.valor = d[key].values[-1]
                if ativo.valor_inicial is None:
                    ativo.valor_inicial = ativo.valor
                #entrada = d[key].values < d[key].values[-1]
                #entrada = numpy.append(entrada, dv[key].values < dv[key].values[-1])
                #entrada = entrada.reshape(len(entrada), 1)
                entrada = geraEntrada(d[key], dv[key])
                saidas[key] = self.rede.feedforward(entrada)
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
            self.patrimonio_medio = sum([ativo.cotas_compradas_se_medio * ativo.valor for ativo in self.ativos])
            self.lista.append(self.patrimonio)
            self.lista_ibov.append(self.df_ibov.values[i])
            self.lista_medio.append(self.patrimonio_medio)
            self.datas.append(date)


        self.dataframe = pd.DataFrame(data = self.lista, index=self.datas, columns=["Adj Close"])
        self.dataframe_medio = pd.DataFrame(data = self.lista_medio, index=self.datas, columns=["Adj Close"])
        plt.figure(1)
        plt.plot(self.dataframe, label="Patrimônio Robo", color="b")
        plt.plot(self.df_ibov, label="Ibovespa", color="r")
        plt.plot(self.dataframe_medio, label="Medio Ativos", color="#00AAAA")
        plt.xlabel('Data')
        plt.ylabel('Patrimônio')
        plt.title('Patrimônio Robo, medio, Ibov')
        plt.legend()
        nome = time.strftime("%Y%m%d%H%M", time.localtime())
        plt.savefig(str(self.id) + "/" + nome + '.png')
        plt.clf()
        plt.figure(2)
        data_rede_td = pd.DataFrame(data=self.rede.td_error, index=self.rede.index, columns=["Error"])
        data_rede = pd.DataFrame(data=self.rede.error, index=self.rede.index, columns=["Error"])
        plt.plot(data_rede_td, label="Erro Test_data", color='r')
        plt.plot(data_rede, label="Erro Trainning_data", color='b')
        plt.xlabel('Temporada')
        plt.ylabel('Erro')
        plt.title('Erros Trainning_data Test_data')
        nome = "Erro" + time.strftime("%Y%m%d%H%M", time.localtime())
        plt.legend()
        plt.savefig(str(self.id) + "/" + nome + '.png')
        plt.clf()
        plt.figure(3)
        for key in self.nomes_ativos:
            plt.plot(self.df[key], label=key)

        plt.legend()
        plt.savefig("Ativos.png")
        #plt.show()
        plt.clf()
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
              "\nGanho do Robô: ", self.patrimonio / CAPITAL_INICIAL,
              "\nGanho medio da ações: ", self.ganho_medio,
              "\nRelação Robô/Ibovespa: ",
              (self.patrimonio / CAPITAL_INICIAL) / (self.ganho_ibov),
              "\nRelação Robô/Ganho medio das ações: ",
              (self.patrimonio / CAPITAL_INICIAL) / (self.ganho_medio),
              "\nCompras: ", self.compras,
              "\nVendas: ", self.vendas,
              "\nTempo médio trade : ", tempomedio, "dias",
              "\nMédia de : ", media, "trades por mês",
              "\nValor_Comprado: ", self.valor_comprado,
              "\nValor_Vendido: ", self.valor_vendido)
