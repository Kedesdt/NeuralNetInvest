import numpy
def geraEntrada(e, edv):

    entrada = [0]

    for ent in range(0, len(e) - 1):
        entrada.append(e[ent] < e[ent + 1])

    entrada2 = [0]
    for ent in range(0, len(edv) - 1):
        entrada2.append(edv[ent] < edv[ent + 1])

    entrada = numpy.append(entrada, entrada2)
    entrada = entrada.reshape(len(entrada), 1)

    return entrada

