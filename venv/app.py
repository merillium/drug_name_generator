import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, Input, Output, callback, dcc

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
    print(df_top.iloc[row][[text_col,'prob']])
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
        print(f"token = {token}")
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

    df_prefix = pd.read_csv('./data/df_prefix.csv').dropna()
    df_middle = pd.read_csv('./data/df_middle.csv').dropna()
    df_suffix = pd.read_csv('./data/df_suffix.csv').dropna()

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
    app.run(debug=False, host="0.0.0.0", port=8080)