import gzip

with open("modelo_entrenado.pkl", "rb") as f_in:
    with gzip.open("modelo_entrenado.pkl.gz", "wb") as f_out:
        f_out.write(f_in.read())

print("✔️ modelo_entrenado.pkl.gz generado correctamente")
