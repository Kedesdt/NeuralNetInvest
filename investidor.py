from ativo import Ativo
from constantes import *
from funcoes import *
import pandas as pd
import matplotlib.pyplot as plt
import time

MIN_COMPRA = 0.75

class Investidor:

    def __init__(self, df, dv, ganho_ibov = 1, df_ibov = None, ID = 1, ativos = None):

        self.id = ID
        self.ativos_treinados = ativos
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
        self.capital_inicial = self.patrimonio
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
                if ativo.valor_anterior:
                    if ativo.comprado:
                        if ativo.valor < ativo.valor_anterior:
                            ativo.erros += 1
                        else:
                            ativo.acertos += 1
                    else:
                        if ativo.valor > ativo.valor_anterior:
                            ativo.erros += 1
                        else:
                            ativo.acertos += 1
                ativo.valor_anterior = ativo.valor
                if ativo.valor_inicial is None:
                    ativo.valor_inicial = ativo.valor

                entrada = geraEntrada_relacao(d[key], dv[key])
                saidas[key] = self.ativo_por_nome(key).rede.feedforward(entrada)
                ativo.valor = d[key].values[-1]
                ativo.predicao = saidas[key][0][0]
                predicoes.append(saidas[key][0][0])

            predicoes.sort()

            if 'nan' in predicoes:
                continue

            ativos_na_ordem = []
            comprar = []
            predicoes_comprar = []

            err = False
            for j in predicoes:
                if j > 1:
                    ativos_na_ordem.append(self.get_by_pre(j))
                    comprar.append(self.get_by_pre(j))
                    predicoes_comprar.append(j)
                    if ativos_na_ordem[-1] is None:
                        err = True
            if err:
                continue
            for j in range(len(self.ativos)):
                self.vende_ativo(self.ativos[j], i)

            #q = QUANTIDADE - QUANTIDADE_DE_ACOES
            q = len(comprar)
            if q > 0:
                for j in comprar:
                    valor = j.predicao * (self.dinheiro_disponivel / sum(predicoes_comprar))
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

        index = 1
        self.dataframe = pd.DataFrame(data = self.lista, index=self.datas, columns=["Adj Close"])
        self.dataframe_medio = pd.DataFrame(data = self.lista_medio, index=self.datas, columns=["Adj Close"])
        plt.figure(index)
        index+=1
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

        for ativo in self.ativos_treinados:
            plt.figure(index)
            index += 1
            data_rede_td = pd.DataFrame(data=ativo.rede.td_error, index=ativo.rede.index, columns=["Error"])
            data_rede = pd.DataFrame(data=ativo.rede.error, index=ativo.rede.index, columns=["Error"])
            plt.plot(data_rede_td, label="Err Tes_d " + ativo.nome)
            plt.plot(data_rede, label="Err Tr_d " + ativo.nome)
            plt.xlabel('Temporada')
            plt.ylabel('Erro')
            plt.title('Erros Trainning_data Test_data')
            nome = "Erro_" + ativo.nome + "_" + time.strftime("%Y%m%d%H%M", time.localtime())
            plt.legend()
            plt.savefig(str(self.id) + "/" + nome + '.png')
            plt.clf()
        plt.figure(index)
        index+=1
        for key in self.nomes_ativos:
            plt.plot(self.df[key], label=key)

        plt.legend()
        plt.savefig("Ativos.png")
        #plt.show()
        plt.clf()
        self.ganho_medio = self.calcula_medio()
        self.print()

        plt.figure(index)
        index += 1

        for ativo in self.ativos_treinados:

            data = pd.DataFrame(data=ativo.acertos[self.id], index=range(len(ativo.acertos[self.id])), columns=["Acertos"])
            plt.plot(data, label="Taxa de acertos " + ativo.nome)

        plt.xlabel('Iteração')
        plt.ylabel('Taxa de acerto')
        plt.title('Taxa de acerto')
        nome = "Taxa_de_acertos"
        plt.legend()
        plt.savefig(str(self.id) + '/' + nome + '.png')

        plt.clf()


    def print(self):

        media = len(self.trades) / (len(self.df) / 30)
        if len(self.trades):
            tempomedio = sum(self.trades) / len(self.trades)
        else:
            tempomedio = 0

        print("Patrimonio Final: ", self.patrimonio,
              "\nPatrimonio Inicial: ", self.capital_inicial,
              "\nGanho do Robô: ", self.patrimonio / self.capital_inicial,
              "\nGanho medio da ações: ", self.ganho_medio,
              "\nRelação Robô/Ibovespa: ",
              (self.patrimonio / self.capital_inicial) / (self.ganho_ibov),
              "\nRelação Robô/Ganho medio das ações: ",
              (self.patrimonio / self.capital_inicial) / (self.ganho_medio),
              "\nCompras: ", self.compras,
              "\nVendas: ", self.vendas,
              "\nTempo médio trade : ", tempomedio, "dias",
              "\nMédia de : ", media, "trades por mês",
              "\nValor_Comprado: ", self.valor_comprado,
              "\nValor_Vendido: ", self.valor_vendido)
        for ativo in self.ativos:
            print("Taxa de acertos ", ativo.nome, " : ", ativo.acertos / (ativo.acertos + ativo.erros))
            self.ativo_por_nome(ativo.nome).acertos[self.id].append(ativo.acertos / (ativo.acertos + ativo.erros))

    def ativo_por_nome(self, nome):
        for ativo in self.ativos_treinados:
            if ativo.nome == nome:
                return ativo

        return None
