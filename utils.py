import pandas as pd

def is_valid_txt(txt: str) -> bool:
    """get text return where True good text"""
    if len(txt.split()) == 0 or pd.isnull(txt) or type(txt) != str:
        return False
    return True