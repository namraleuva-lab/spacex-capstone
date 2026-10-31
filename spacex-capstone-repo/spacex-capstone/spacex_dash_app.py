import pandas as pd, dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

df = pd.read_csv("data/spacex_launch_dash.csv")
df["Launch Site"] = df["Launch Site"].replace({"CCAFS LC-40": "CCAFS SLC-40"})
lo, hi = df["Payload Mass (kg)"].min(), df["Payload Mass (kg)"].max()
sites = [{"label": "All Sites", "value": "ALL"}] + [{"label": s, "value": s} for s in sorted(df["Launch Site"].unique())]

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("SpaceX Launch Records Dashboard", style={"textAlign": "center"}),
    dcc.Dropdown(id="site-dropdown", options=sites, value="ALL", placeholder="Select a Launch Site", searchable=True),
    dcc.Graph(id="success-pie-chart"),
    html.P("Payload range (kg):"),
    dcc.RangeSlider(id="payload-slider", min=0, max=10000, step=1000, value=[lo, hi]),
    dcc.Graph(id="success-payload-scatter-chart"),
])

@app.callback(Output("success-pie-chart", "figure"), Input("site-dropdown", "value"))
def pie(site):
    if site == "ALL":
        return px.pie(df, values="class", names="Launch Site", title="Total Successful Launches by Site")
    sub = df[df["Launch Site"] == site]["class"].value_counts().reset_index()
    sub.columns = ["class", "count"]
    return px.pie(sub, values="count", names="class", title=f"Success (1) vs Failure (0) for {site}")

@app.callback(Output("success-payload-scatter-chart", "figure"),
              [Input("site-dropdown", "value"), Input("payload-slider", "value")])
def scatter(site, rng):
    sub = df[df["Payload Mass (kg)"].between(rng[0], rng[1])]
    if site != "ALL":
        sub = sub[sub["Launch Site"] == site]
    return px.scatter(sub, x="Payload Mass (kg)", y="class", color="Booster Version Category",
                      title="Payload vs Launch Outcome")

if __name__ == "__main__":
    app.run()
