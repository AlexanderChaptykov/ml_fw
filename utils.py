import pandas as pd

def is_valid_txt(txt: str) -> bool:
    """get text return where True good text"""
    if type(txt) != str:
        return False
    if len(txt.split()) == 0 or pd.isnull(txt):
        return False
    return True