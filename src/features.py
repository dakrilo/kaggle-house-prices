def add_features(df):
    """
    Добавляет новые признаки домов, посчитанные из существующих.
    Старые столбцы не меняются. Возвращает копию таблицы.
    """
    df = df.copy()

    # Суммы: несколько связанных площадей / счётчиков → одно число

    # Вся площадь дома: подвал + 1-й этаж + 2-й этаж
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # Всего ванных. В данных два вида (см. data_description.txt):
    # FullBath, BsmtFullBath: полная ванная (душ или ванна, туалет, раковина);
    # HalfBath, BsmtHalfBath: «половинка», только туалет и раковина, без душа.
    # Половинку считаем за 0.5, как в объявлениях о продаже в США («2.5 baths»)
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Вся «уличная» площадь: четыре типа веранд + деревянная терраса
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    # Возраст: год события → сколько лет прошло к моменту продажи
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # Флаги: количество / площадь > 0 → 1 (есть), иначе 0 (нет)
    df["HasGarage"] = (df["GarageCars"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    # Взаимодействие: большой и качественный дом дороже, чем сумма двух вкладов
    df["QualxArea"] = df["OverallQual"] * df["GrLivArea"]

    # Нелинейность: каждая следующая ступень качества добавляет к цене больше
    df["Qual2"] = df["OverallQual"] ** 2

    return df
