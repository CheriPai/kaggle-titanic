def clean_data(df):

    # Drop irrelevant data
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    # Replace categorical values with numerical values
    df.Sex = df.Sex.replace(['male', 'female'], [0, 1])
    df.Embarked = df.Embarked.replace(['S', 'C', 'Q'], [0, 1, 2])

    # Replace missing values with median
    for col in df:
        df[col] = df[col].fillna(df[col].median())

    return df
