def add_features(df):
    """Добавляет новые признаки. Возвращает копию таблицы."""
    df = df.copy()

    title = df["Name"].str.extract(r",\s*([^\.]+)\.")[0]
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = title.where(title.isin(["Mr", "Miss", "Mrs", "Master"]), "Rare")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    return df
