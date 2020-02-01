import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import webbrowser
import datetime


from configs import metrics, weeks, lstm_features, target
from prediction import Prediction


def create_dashboard(prediction):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] #internet access to get .css file
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    days, hours, locations = metrics['days'], metrics['hours'], metrics['locations'],
    app.layout = html.Div([
        html.Div([html.H1(" Real Time Mobile App Retantion And Promotion Open - Close Deciding")],
                 style={'textAlign': "left", "padding-bottom": "10", "padding-top": "10"}),
        html.Div(
            [html.Div(dcc.Dropdown(id="days",
                                   options=[{'label': i, 'value': i} for i in days],
                                   value=1, ), className="four columns",
                      style={"display": "block", "margin-left": "!%",
                             "margin-right": "auto", "width": "24%"}),
             html.Div(dcc.Dropdown(id="hours",
                                   options=[{'label': i, 'value': i} for i in hours],
                                   value=20, ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "auto", "width": "24%"}),
             html.Div(dcc.Dropdown(id="locations",
                                   options=[{'label': i, 'value': i} for i in locations],
                                   value=locations[0]), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "1%", "width": "24%"}),
             html.Div(dcc.Dropdown(id="weeks",
                                   options=[{'label': i, 'value': i} for i in weeks],
                                   value=weeks[-1]), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "1%", "width": "24%"})
             ], className="row", style={"padding": 14, "display": "block", "margin-left": "1%",
                                        "margin-right": "auto", "width": "99%"}),

        # Graphs
        html.Div(
            [html.Div([
                ## TODO: shows minutelly funnel lie chart for each feature
                dcc.Graph(id="funnel", hoverData={'points': [{'PaymentTransactionId': 21903610}]})],
                      className="row", style={"padding": 0, "display": "inline-block", "width": "100%"}),
             html.Div([
                 ## TODO: shows close open ratio bar chart
                 dcc.Graph(id="last-30-mins-close-open"),
                 ## TODO: show prediction values of each metric with its actual value
                 dcc.Graph(id="last-30-minprediction-comperation")],
                      className="row", style={"padding": 50, "display": "inline-block", "width": "100%"})
             ], className="row", style={"padding": 0, "display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "100%"}),
        dcc.Interval(
                id='graph-update',
                interval=100 * 10000000
            )
    ], style={"margin-left": "1%"})


    @app.callback(
        dash.dependencies.Output('funnel', 'figure'),
        [dash.dependencies.Input('days', 'value'),
         dash.dependencies.Input('hours', 'value'),
         dash.dependencies.Input('locations', 'value'),
         dash.dependencies.Input('weeks', 'value'),
         dash.dependencies.Input("graph-update", "n_intervals")
        ]
    )
    def update_graph(day, hour, location, week, n):
        min = datetime.datetime.now().minute if datetime.datetime.now().minute != 60 else 0
        prediction.get_prediction_values(day, hour, location, week, min)
        prediction.data = prediction.data[prediction.data['index'] < min]
        trace = []
        for f in lstm_features:
            _data = go.Scatter(
                                x=prediction.data['date'],
                                y=prediction.data[f],
                                text=prediction.data[f],
                                customdata=prediction.data[f],
                                mode='lines+markers',
                                name=f + ' Retention' if f != 'ratios' else 'System Availability Retention',
                                marker={
                                    'size': 15,
                                    'opacity': 0.5,
                                    'line': {'width': 0.5, 'color': 'white'}
                                }
                )
            trace.append(_data)

        return {
            'data': trace,
            'layout': go.Layout(
                yaxis={
                    'title': 'Retention'
                },
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450,
                hovermode='closest',
                legend_orientation="h"
            )
        }

    # Bar Chart
    @app.callback(
        dash.dependencies.Output('last-30-mins-close-open', 'figure'),
        [dash.dependencies.Input('days', 'value'),
         dash.dependencies.Input('hours', 'value'),
         dash.dependencies.Input('locations', 'value'),
         dash.dependencies.Input('weeks', 'value'),
         dash.dependencies.Input("graph-update", "n_intervals")
        ]
    )

    def update_graph_2(day, hour, location, week, n):
        print("okadkaosdkaosdoaskdoasdk")
        print(prediction.data.head())
        min = datetime.datetime.now().minute if datetime.datetime.now().minute != 60 else 0
        prediction.get_prediction_values(day, hour, location, week, min)
        prediction.data = prediction.data[prediction.data['index'] < min]
        print("oookkkkkkksssdasslmafmwl")
        open = len(prediction.data.query("close == 0"))
        close = len(prediction.data.query("close == 1"))
        print("adasdasdasdsdadads!!!!!!!!")
        print(open, close)
        return {"data": [
            go.Bar(name='Closing Promotion Count', x=[1], y= [close]),
            go.Bar(name='Opening Promotion Count', x=[0], y= [open])],
                "layout": go.Layout(height=300,
                                    title="sfdgsfchg",
                                    )
                }

    webbrowser.open('http://127.0.0.1:8050/')
    app.run_server(debug=False, port=8050, host='127.0.0.1')












