import os


PACIENTE = "./resultados/paciente"
CONTROLE = "./resultados/controle"


def rename_files(root, dirs):
    for dir in dirs:

        person_path = os.path.join(root, dir)
        print()
        for window in os.listdir(person_path):
            full_path = os.path.join(person_path, window)

            if(not ("original.wav" in os.listdir(full_path)) or not("product.wav" in os.listdir(full_path))):
                print("One of the files was not found: original.wav, product.wav")
                continue

            person, window = full_path.split("/")[-2:]

            person_number = person.split("-")[-1]
            window_number = window.split("-")[-1]
            person_group = "p" if "paciente" in person else "c"

            new_file_id = "{}{}w{}".format(
                person_group, person_number, window_number)

            old_original = os.path.join(full_path, "original.wav")
            new_original = os.path.join(
                full_path, "{}-original.wav".format(new_file_id))
            old_product = os.path.join(full_path, "product.wav")
            new_product = os.path.join(
                full_path, "{}-product.wav".format(new_file_id))

            print(old_original, "->", new_original)
            print(old_product, "->", new_product)
            os.rename(old_original, new_original)
            os.rename(old_product, new_product)


rename_files(PACIENTE, os.listdir(PACIENTE))
rename_files(CONTROLE, os.listdir(CONTROLE))
