with open("../tinder/tinder.csv","r") as tinder_csv:
	tinder = [l.split(",")[2] for l in tinder_csv.readlines()[1:]]
	tinder = [l.replace(".opus",".wav") for l in tinder]
	tinder = [l.split("/")[1] for l in tinder]


with open("/data/SPIRA_Dataset_V2/metadata_eval.csv","r") as validacao_csv:
	validacao = [l.split(",")[0] for l in validacao_csv.readlines()[1:]]
	validacao = [l.replace(".opus",".wav") for l in validacao]
	validacao = [l.split("/")[1] for l in validacao]

#print(validacao)


for linha in tinder:
	if(linha in validacao):
		print(linha)
