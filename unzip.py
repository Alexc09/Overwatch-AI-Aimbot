import tarfile, glob

for file in glob.glob("*.tar.gz"):
    print(file)
    cont = input("Proceed to unzip file?:")
    if cont == 'y':
        if file.endswith("tar.gz"):
            tar = tarfile.open(file, "r:gz")
            tar.extractall()
            tar.close()

        elif file.endswith("tar"):
            tar = tarfile.open(file, "r:")
            tar.extractall()
            tar.close()