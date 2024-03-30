with open("../tinder/tinder.csv","r") as tinder_csv:
	tinder = [l.split(",")[2] for l in tinder_csv.readlines()[1:]]
	tinder = [l.replace(".opus",".wav") for l in tinder]
	tinder = [l.split("/")[1] for l in tinder]


with open("/data/SPIRA_Dataset_V2/metadata_test.csv","r") as teste_csv:
	teste = [l.split(",")[0] for l in teste_csv.readlines()[1:]]
	teste = [l.replace(".opus",".wav") for l in teste]
	teste = [l.split("/")[1] for l in teste]

#print(validacao)


for linha in tinder:
	if(linha in teste):
		print(linha)
