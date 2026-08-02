from sklearn.model_selection import train_test_split


def split_subjects(
    subjects,
    test_size=0.2,
    random_state=42,
):
    """
    Split OCT subjects into train and validation groups.

    Important:
    Splits happen at subject level to avoid
    patient leakage.
    """

    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=random_state,
    )

    return train_subjects, val_subjects