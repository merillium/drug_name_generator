from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, Input, Output, callback, dcc

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

def get_random_token(df, text_col, top=None, seed=None):
    """
    Get a random token and its probability from a two-column df
    """
    if seed is not None:
        np.random.seed(seed)
    
    if top is not None:
        df_top = df.iloc[:top].copy()
    else:
        df_top = df.copy()
    df_top['prob'] = df_top['count'] / df_top['count'].sum()
    size = len(df_top)
    indices = np.array(range(size))
    p = df_top['prob'].to_numpy()
    row = np.random.choice(a=indices, p=p)
    (token, prob) = df_top.iloc[row][[text_col,'prob']]
    return (token, prob)

def generate_drug_name(dfs, text_col="vowel_token", top=100, seed=None):
    """
    Generate a drug name by concatenating a prefix, middle, and suffix.
    The score is the product of their probabilities (frequency)

    Parameters
    ----------
    dfs: List[pd.DataFrame]
        a list of df (prefix, middle, suffix),
        each having columns {text_col}, 'count' ordered by descending count

    """
    total_score = 1
    drug_name = ""
    for df in dfs:
        token, prob = get_random_token(df.iloc[:top], text_col, seed)
        drug_name += token
        total_score *= prob 
    return (drug_name, total_score)

app = Dash()
app.layout = html.Div([
    dcc.Graph(id='drugname-scatter', figure=go.Figure()),
    html.Div([
        html.Button(id='refresh', n_clicks=0, children='Regenerate Drug Names'),
        html.Br(),
        html.Label("Temperature (randomness)"),
        dcc.Slider(id='temperature', min=0, max=100, marks=None, value=10, tooltip={"placement": "bottom", "always_visible": True}, className = 'slider'),
        html.Label("Number of drug names"),
        dcc.Slider(id='newDrugCount', min=5, max=15, marks=None, value=10, step=1, tooltip={"placement": "bottom", "always_visible": True}, className = 'slider'),
    ],style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
    })

])

@callback(
    Output('drugname-scatter', 'figure'),
    Input('refresh', 'n_clicks'),
    Input('temperature', 'value'),
    Input('newDrugCount', 'value'),
)
def create_drug_names_figure(n_clicks, temperature, drug_count):
    """
    Creates a word cloud scatter plot of 10 randomly generated drug names 
    where the size of the word is based on the probability score of the drug.
    Drugs with higher scores (using more frequently observed prefix, middle, suffix)
    will appear larger in the word cloud. Font size is scaled to between 10 and 20

    Parameters
    ----------
    n_clicks: int
        placeholder for the refresh button click
    temperature: int
        randomness where minimum temperature corresponds to picking top 100 choices,
        and maximum temperature corresponds to picking top 500 choices. since these are still based
        on frequency, top 500 choices may not result in significantly different outcomes,
        but could theoretically pick less frequently occurring prefix, middle, suffix
    drug_count: int
        number of drugs to generate, minimum = 5 and maximum = 15, defaults to 10 drugs

    """
    drug_count = int(drug_count)
    all_drug_names = []
    all_scores = []

    top = int((500-100)/(100-10)*(temperature-10) + 100)
    for _ in range(int(drug_count)):
        # print(f"pulling tokens from top {top} entries")
        drug_name, score = generate_drug_name(dfs=[df_prefix,df_middle,df_suffix], top=top)
        all_drug_names.append(drug_name)
        all_scores.append(score)
        # print(f"drug_name {drug_name} with score = {score}")

    all_scores = np.array(all_scores)

    ## rescale scores such that minimum score has a font size of 20, max score has a font size of 20
    a,b = 10,20
    font_sizes = (b - a) * (all_scores - min(all_scores)) / (max(all_scores) - min(all_scores)) + a

    xs = np.random.rand(drug_count)
    ys = np.random.rand(drug_count)
    colors = px.colors.qualitative.Plotly[:drug_count]
    extra_colors = px.colors.qualitative.Prism
    if len(colors) < drug_count:
        diff = drug_count - len(colors)
        more_colors = extra_colors[:diff]
        colors.extend(more_colors)

    fig = go.Figure()
    for x,y,drug_name,font_size,color in zip(xs,ys,all_drug_names,font_sizes,colors):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            text=[drug_name],
            mode='text',
            textfont=dict(size=font_size, color=color),
            showlegend=False,
        ))
    fig.update_layout(
        template='simple_white', 
        xaxis={'range': [-0.2,1.2], 'visible': False}, 
        yaxis={'range': [-0.2,1.2], 'visible': False},  
    )
    fig.update_traces(hovertemplate='%{text}<extra></extra>')
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)