def clean_data(df):
    """Cleans raw data to pass into classifier"""

    # Create new title feature
    titles = df.Name.apply(get_title)
    for k, v in title_mapping.items():
        titles[titles == k] = v
    for i, title in enumerate(titles):
        if type(title) is not int:
            titles[i] = 0
    df['Title'] = titles

    # Drop irrelevant features
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    # Replace categorical values with numerical values
    df.Sex = df.Sex.replace(['male', 'female'], [0, 1])
    df.Embarked = df.Embarked.replace(['S', 'C', 'Q'], [0, 1, 2])

    # Replace missing values with median
    for col in df:
        df[col] = df[col].fillna(df[col].median())

    return df


def get_title(name):
    """Extracts the the title of the passenger from their name"""
    return [x for x in name.split() if x[-1] == '.'][0]


title_mapping = {
    "Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Dr.": 5, "Rev.": 6, 
    "Major.": 7, "Col.": 7, "Mlle.": 8, "Mme.": 8, "Don.": 9, "Lady.": 10, 
    "Countess.": 10, "Jonkheer.": 10, "Sir.": 9, "Capt.": 7, "Ms.": 2
}
