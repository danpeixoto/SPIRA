#! /usr/bin/python3

import csv
from os import listdir
from os.path import isfile, join


CAMINHO_PACIENTES ="/data/Tinder/paciente/"
CAMINHO_CONTROLE = "/data/Tinder/controle/"

#print(listdir(CAMINHO_PACIENTES+"202021"))

pasta_pacientes = [d for d in listdir(CAMINHO_PACIENTES)]

#print(join(CAMINHO_PACIENTES,pasta_pacientes[0]))

arquivos_pacientes = [d+"/"+f.replace(".opus",".wav") for d in pasta_pacientes for f in listdir(join(CAMINHO_PACIENTES,d)) if isfile(join(join(CAMINHO_PACIENTES,d),f))]
arquivos_controle = ["controle/"+f for f in listdir(CAMINHO_CONTROLE) if isfile(join(CAMINHO_CONTROLE, f))]

#print(arquivos_pacientes)


with open("../tinder/tinder.csv","r") as csv:
	tinder_csv = [l.split(",")[2] for l in csv.readlines()[1:]]
	tinder = [l.replace(".opus",".wav") for l in tinder_csv]


flag = True
arquivos_presentes = []

for linha in tinder:
	if((linha in arquivos_pacientes) or (linha in arquivos_controle)):
		arquivos_presentes.append(linha)
	else:
		flag = False
		print(linha)




