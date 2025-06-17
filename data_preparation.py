
import pandas as pd
import numpy as np
import os

def check_and_confirm(filepath):
    if os.path.exists(filepath):
        response = input(f"⚠️ Le fichier '{filepath}' existe déjà. Voulez-vous le remplacer ? (o/n) : ")
        if response.lower() != 'o':
            print("⛔ Opération annulée.")
            exit()

def prepare_data(x_csv_path, y_csv_path, random_state=42, train_ratio=0.8, output_prefix=""):
    X = pd.read_csv(x_csv_path)
    y = pd.read_csv(y_csv_path)

    X.insert(0, 'teta0', 1)

    data = pd.concat([X, y], axis=1)
    shuffled = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_index = int(len(shuffled) * train_ratio)
    train = shuffled.iloc[:split_index]
    test = shuffled.iloc[split_index:]

    X_train = train.iloc[:, :X.shape[1]]
    y_train = train.iloc[:, X.shape[1]:]
    X_test = test.iloc[:, :X.shape[1]]
    y_test = test.iloc[:, X.shape[1]:]

    # Vérification avant d'écraser
    for name in ["x_train.csv", "y_train.csv", "x_test.csv", "y_test.csv",
                 "x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]:
        check_and_confirm(f"{output_prefix}{name}")

    # Sauvegarde CSV
    X_train.to_csv(f"{output_prefix}x_train.csv", index=False)
    y_train.to_csv(f"{output_prefix}y_train.csv", index=False)
    X_test.to_csv(f"{output_prefix}x_test.csv", index=False)
    y_test.to_csv(f"{output_prefix}y_test.csv", index=False)

    # Sauvegarde NPY
    np.save(f"{output_prefix}x_train.npy", X_train.to_numpy())
    np.save(f"{output_prefix}y_train.npy", y_train.to_numpy())
    np.save(f"{output_prefix}x_test.npy", X_test.to_numpy())
    np.save(f"{output_prefix}y_test.npy", y_test.to_numpy())

    print("✅ Données préparées et fichiers enregistrés.")

if __name__ == "__main__":
    x_path = "X_standardized.csv"
    y_path = "y_target.csv"
    prepare_data(x_path, y_path, random_state=42, train_ratio=0.8, output_prefix="")
