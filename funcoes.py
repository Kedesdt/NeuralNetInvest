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

def geraEntrada_relacao(e, edv):

    entrada = [0]

    for ent in range(0, len(e) - 1):

        ultimo = e[-1]
        if ultimo:
            entrada.append(e[ent] / ultimo)
        else:
            entrada.append(0)

    entrada2 = [0]
    for ent in range(0, len(edv) - 1):
        ultimo = edv[-1]
        if ultimo:

            entrada2.append(edv[ent] / ultimo)
        else:
            entrada.append(0)

    entrada = numpy.append(entrada, entrada2)
    entrada = entrada.reshape(len(entrada), 1)

    tem_nan = numpy.isnan(entrada).any()
    tem_inf = numpy.isinf(entrada).any()

    if tem_nan or tem_inf:
        print("NAN ou INF")

    return entrada
