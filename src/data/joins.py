import pandas as pd

def build_member_policy_base(members: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    df = members.merge(
        policies,
        on=["member_id", "policy_id"],
        how="left",
        validate="one_to_one"
    )
    return df
