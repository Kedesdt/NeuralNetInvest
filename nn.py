"""
network1.py
~~~~~~~~~~

Obs: Este script é baseado na versão do livro http://neuralnetworksanddeeplearning.com/, com a devida autorização do autor.

Um módulo para implementar uma rede neural com o aprendizado baseado no algoritmo Stochastic Gradient Descent para uma rede neural feedforward. 
Os gradientes são calculados usando backpropagation. 
Note que este é um código simples, facilmente legível e facilmente modificável. 
Não é otimizado e omite muitos recursos desejáveis. 
O objetivo aqui é compreender bem os conceitos fundamentais e alguns conceitos mais avançados serão discutidos nos próximos capítulos do livro.

"""

# Imports
import random
import numpy as np

# Classe Network
class Network(object):

    def __init__(self, sizes):
        """A lista `sizes` contém o número de neurônios nas
         respectivas camadas da rede. Por exemplo, se a lista
         for [2, 3, 1] então será uma rede de três camadas, com o
         primeira camada contendo 2 neurônios, a segunda camada 3 neurônios,
         e a terceira camada 1 neurônio. Os bias e pesos para a
         rede são inicializados aleatoriamente, usando uma distribuição Gaussiana com média 0 e variância 1. 
         Note que a primeira camada é assumida como uma camada de entrada, e por convenção nós
         não definimos nenhum bias para esses neurônios, pois os bias são usados
         na computação das saídas das camadas posteriores."""

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.errup = False
        self.contup = 0
        self.maxerrup = 20000
        self.lasterr = 0

    def feedforward(self, a):
        """Retorna a saída da rede se `a` for input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Treinar a rede neural usando mini-batch stochastic
        gradient descent. O `training_data` é uma lista de tuplas
         `(x, y)` representando as entradas de treinamento e as
         saídas. Os outros parâmetros não opcionais são
         auto-explicativos. Se `test_data` for fornecido, então a
         rede será avaliada em relação aos dados do teste após cada
         época e progresso parcial impresso. Isso é útil para
         acompanhar o progresso, mas retarda as coisas substancialmente."""

        training_data = list(training_data)
        n = len(training_data)

        l = int(len(training_data) * 0.2/0.8)
        random.shuffle(training_data)
        td2 = training_data[-l-1:]

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            if self.errup:
                print("Erro aumentando")
                break
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:


                if j % 1 == 0:
                    print("Epoch {} : Erro: {} / {}".format(j,self.evaluate(test_data),n_test), end= " ")
                else:
                    print("Epoch {} finalizada".format(j))
                print("Erro 2: {} / {}".format(self.evaluate(td2), len(td2)))
            else:
                print("Epoch {} finalizada".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Atualiza os pesos e bias da rede aplicando
         a descida do gradiente usando backpropagation para um único mini lote.
         O `mini_batch` é uma lista de tuplas `(x, y)`, e `eta` é a taxa de aprendizado."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Retorna uma tupla `(nabla_b, nabla_w)` representando o
         gradiente para a função de custo C_x. `nabla_b` e
         `nabla_w` são listas de camadas de matrizes numpy, semelhantes
         a `self.biases` e `self.weights`."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = x

        # Lista para armazenar todas as ativações, camada por camada
        activations = [x] 

        # Lista para armazenar todos os vetores z, camada por camada
        zs = [] 

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Aqui, l = 1 significa a última camada de neurônios, l = 2 é a 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Retorna o número de entradas de teste para as quais a rede neural 
         produz o resultado correto. Note que a saída da rede neural
         é considerada o índice de qualquer que seja
         neurônio na camada final que tenha a maior ativação."""

        random.shuffle(test_data)
        e = 0
        for (x, y) in test_data:

            r = self.feedforward(x)

            #if e==0:
            #    print(r)
            #    print(y)

            for i in range(len(y)):
                e += abs(r[i][0] - y[i][0])

        if self.lasterr < e:
            self.contup += 1
        else:
            self.contup = 0

        self.errup = self.contup >= self.maxerrup
        self.lasterr = e

        #test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        #return sum(int(x == y) for (x, y) in test_results)
        return e / len(y)

    def cost_derivative(self, output_activations, y):
        """Retorna o vetor das derivadas parciais."""
        return (output_activations-y)

    def clone(self):

        novo = Network(self.sizes)

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                novo.biases[i][j] = self.biases[i][j]

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    novo.weights[i][j][k] = self.weights[i][j][k]

        return novo


    def muta(self, taxa):

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                if random.random() < taxa:
                    if random.random() < taxa / 2:
                        self.biases[i][j] = (random.random()- 0.5) * 2
                    else:
                        self.biases[i][j] += (random.random() - 0.5) * taxa * 0.1
                        if self.biases[i][j] > 1:
                            self.biases[i][j] = 1
                        if self.biases[i][j] < - 1:
                            self.biases[i][j] = - 1

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if random.random() < taxa:
                        if random.random() < taxa / 2:
                            self.weights[i][j][k] = (random.random()- 0.5) * 2
                        else:
                            self.weights[i][j][k] += (random.random() - 0.5) * taxa * 0.1
                            if self.weights[i][j][k] > 1:
                                self.weights[i][j][k] = 1
                            if self.weights[i][j][k] < - 1:
                                self.weights[i][j][k] = - 1


# Função de Ativação Sigmóide
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
