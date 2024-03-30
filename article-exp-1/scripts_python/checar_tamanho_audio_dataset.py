import torchaudio
import pandas as pd


df = pd.read_csv("../tinder/tinder_sem_audios_pequenos.csv")
#print(df.head())

def checar_tamanho(arquivo):
	if(".opus" in arquivo):
		arquivo = "/data/Tinder/paciente/"+arquivo
	else:
		arquivo = "/data/Tinder/"+arquivo

	audio,sr = torchaudio.load(arquivo, normalization=True)
	#print(sr)
	#print(audio.shape,sr)
	if(audio.shape[1]/sr < 4.5):
        	print("Audio: {} com tamanho {}".format(arquivo, audio.shape[1]/sr))


df.apply(lambda x: checar_tamanho(x["arquivo"]), axis=1)



