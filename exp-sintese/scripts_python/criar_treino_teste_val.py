from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("../tinder/tinder_sem_audios_pequenos.csv")
df =  df.drop(df.columns[0], axis=1) #drop a coluna de indices do tinder
df =  df.drop(df.columns[0], axis=1) #drop a coluna tupla do tinder


#preciso ordernar as colunas para que o código aceite
#valores corretos [arquivo,classe,sexo,idade,...]
#valor df ['arquivo', 'sexo', 'idade', 'oxigenacao', 'base', 'done']
df_cols = ['arquivo','base','sexo','idade','oxigenacao','done']
print(df_cols)
df = df[df_cols]

#transforma os dados categóricos para numéricos
df.loc[df["base"]=="P","base"] = 1
df.loc[df["base"]=="C","base"] = 0


# não precisou dessa parte, mas vou deixar aqui caso algum dia precise
#--------------------------transforma todos .opus em .wav
#--------------------------df['arquivo'] = df['arquivo'].apply(lambda x: x.replace('.opus', '.wav'))



#adiciona paciente ao caminho do audio

def verificar_arquivo(arquivo):
	return arquivo if "controle" in arquivo else "paciente/"+arquivo 

#print(df["arquivo"].str.contains("controle") == False)

df["arquivo"] = df.apply(lambda x: verificar_arquivo(x["arquivo"]), axis=1)



print(df.head())



# Criar a separação do df

treino, teste = train_test_split(df, test_size=0.2, random_state = 42)
treino, validacao = train_test_split(treino, test_size=0.1, random_state =42)

#print(validacao)


teste.to_csv("../tinder/metadata_teste.csv", encoding='utf-8', index=False)
treino.to_csv("../tinder/metadata_treino.csv", encoding='utf-8', index=False)
validacao.to_csv("../tinder/metadata_eval.csv", encoding='utf-8', index=False)
