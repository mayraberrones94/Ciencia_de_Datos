def procesa(line):
    salida = ''
    ignora = False
    for c in line:
        if c == '"':
            ignora = not ignora
            continue
        else:
            if c == '\n':
                return salida
            if not ignora:
                salida += c
            elif not c == ',':
                salida += c
    return salida

from sys import argv
with open(argv[1], 'r') as input:
    for line in input:
        print(procesa(line))