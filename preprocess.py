from collections import Counter
import pandas as pd

## source of txt file: https://www.fda.gov/media/89850/download?attachment
df = pd.read_csv('./data/product.txt', sep="\t", encoding='unicode_escape')

train_df = df.loc[df['PRODUCTTYPENAME'] == 'HUMAN PRESCRIPTION DRUG',['PROPRIETARYNAME']].copy()
train_df = train_df.drop_duplicates()
train_df = train_df.rename(columns={'PROPRIETARYNAME': 'name'})

## drop any drugs that include numbers in the name, such as "50MG STELARA"
train_df['name'] = train_df['name'].str.findall(r'\b[A-Za-z]+\b').str.join(' ').str.upper().astype(str)

## drop any drugs that are multiple words so that we don't include names like "STERILE DILUENT"
filtered_df = train_df[~train_df['name'].str.contains(" ")].copy()

## this is effectively a corpus of 6,986 drug name "words"

## split by tokens ending in 1-2 vowels (AEIOUY)
## preferring two consecutive vowels

## EXAMPLES: 
## (1) wegovy --> we + go + vy
## (2) zepbound --> ze + pbou + nd
## (3) ziana --> zia + na
## (4) amyvid --> a + my + vi + d

vowel_pattern = r'[^AEIOUY]*[AEIOUY]{2}|[^AEIOUY]*[AEIOUY]|[^AEIOUY]+$'
filtered_df['vowel_tokens'] = filtered_df['name'].str.findall(vowel_pattern)

def combine_vowel_tokens(row_list):
    """Combine single letter splits at the beginning or end 
    to create a truer prefix or suffix

    Examples: 
    [A, MY, VI, D] --> [AMY, VID]
    [A, FLO, RA] --> [AFLO, RA]
    [ZI, ANA] --> [ZI, ANA] 
    """

    ## create suffix
    if len(row_list[-1]) == 1:
        row_list = row_list[:-2] + [row_list[-2] + row_list[-1]]

    ## create prefix
    if len(row_list[0]) == 1:
        row_list = [row_list[0] + row_list[1]] + row_list[2:]
    
    return row_list

filtered_df['vowel_tokens_compact'] = filtered_df['vowel_tokens'].apply(lambda row: combine_vowel_tokens(row))
filtered_df['prefix'] = filtered_df['vowel_tokens_compact'].apply(lambda row: [row[0]])
filtered_df['middle'] = filtered_df['vowel_tokens_compact'].apply(lambda row: row[1:-1])
filtered_df['suffix'] = filtered_df['vowel_tokens_compact'].apply(lambda row: [row[-1]])

## extract list of all strings
def extract_tokens(df, col):
    xss = [item for sublist in df[col].tolist() for item in sublist]
    tokens_counts = Counter(xss)
    df_tokens = pd.DataFrame(
        dict(vowel_token=tokens_counts.keys(), count=tokens_counts.values()
    )).sort_values(by='count', ascending=False)
    return df_tokens

df_prefix = extract_tokens(filtered_df, 'prefix')
df_middle = extract_tokens(filtered_df, 'middle')
df_suffix = extract_tokens(filtered_df, 'suffix')

df_prefix.to_csv('./venv/data/df_prefix.csv', index=False)
df_middle.to_csv('./venv/data/df_middle.csv', index=False)
df_suffix.to_csv('./venv/data/df_suffix.csv', index=False)