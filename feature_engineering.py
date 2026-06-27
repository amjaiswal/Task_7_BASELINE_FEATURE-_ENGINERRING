import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_engineer_features(file_path):
    """
    Load dataset, clean data, perform feature engineering,
    encode categorical variables, and return processed features.
    """

    try:
        # Load dataset
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")

        # Remove duplicate rows
        df = df.drop_duplicates()
        print("Duplicate rows removed.")

        # Fill missing values in numeric columns
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        # Fill missing values in categorical columns
        categorical_columns = df.select_dtypes(include=["object"]).columns
        df[categorical_columns] = df[categorical_columns].fillna("Unknown")

        print("Missing values handled.")

        # -------------------------------
        # Feature Engineering
        # -------------------------------

        # Annual Income
        df["IncomePerYear"] = df["MonthlyIncome"] * 12

        # Experience Ratio
        df["ExperienceRatio"] = df["YearsAtCompany"] / (df["Age"] + 1)

        # Income per Experience
        df["IncomePerExperience"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

        # Promotion Gap
        df["PromotionGap"] = (
            df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
        )

        print("Feature engineering completed.")

        # -------------------------------
        # Remove unnecessary columns
        # -------------------------------

        columns_to_drop = [
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours",
            "Over18"
        ]

        df.drop(columns=columns_to_drop, inplace=True)

        print("Unnecessary columns removed.")

        # -------------------------------
        # Encode categorical variables
        # -------------------------------

        encoder = LabelEncoder()

        for column in df.select_dtypes(include="object").columns:
            df[column] = encoder.fit_transform(df[column])

        print("Categorical features encoded.")

        # -------------------------------
        # Split Features and Target
        # -------------------------------

        X = df.drop("Attrition", axis=1)
        y = df["Attrition"]

        print("Features and target separated.")

        return X, y

    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None, None

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None, None