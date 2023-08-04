from flask import Flask, render_template, request, jsonify, session
import pymate
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter
import matplotlib.figure as Figure
from matplotlib import pyplot as plt
import base64
from io import BytesIO
import logging
import random as rd
import scipy
import statistics
from flask_sqlalchemy import SQLAlchemy
import click
import numpy as np
import time
import json
import pandas as pd

plt.switch_backend('Agg')

app = Flask(__name__)
app.secret_key = 'key_that_is_secret'

app.logger.setLevel(logging.DEBUG)
app.logger.addHandler(logging.StreamHandler())

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://username:password@localhost/database_name'
# db = SQLAlchemy(app)

# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(50), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)

#     def __repr__(self):
#         return '<User %r>' % self.username
    
# @app.cli.command()
# def initdb():
#     """Initialize the database."""
#     db.create_all()
#     click.echo('Database tables created.')

@app.route('/', methods=['GET', 'POST'])
def home():
     global model
     global result
    
     model = None
     result = ''

     return render_template('index.html')

@app.route('/set-up-model', methods=['POST'])
def set_up_model():
    rankCompetitivenessCorrelation = float(request.form['rankCompetitivenessCorrelation'])
    nMales = int(request.form['nMales'])
    nFemales = int(request.form['nFemales'])
    nGroups = int(request.form['nGroups'])
    nGenerations = int(request.form['nGenerations'])

    global model
    model = pymate.evolvingModel(rankCompetitivenessCorrelation=rankCompetitivenessCorrelation, nMales=nMales, nFemales=nFemales, nGroups=nGroups,
                                 nGenerations = nGenerations)
    return ''

@app.route('/go-one-generation', methods=['POST'])
def go_one_generation():
    global model

    if (model is not None) & (model.generation < model.nGenerations):
        for g in model.groups:
            g.runModel()
        
        for g in model.groups:
            g.setupNextGen()
            g.setGenotypes()
            g.generation += 1
        if model.dispersal == True:
                model.migration()

        model.generation += 1
        stop = str(model.generation >= model.nGenerations)

    return stop

@app.route('/go-one-day', methods=['POST'])
def go_one_day():
    global model

    if model is not None:
        for g in model.groups:
            g.females = sorted(g.females, key=g.sortSwelling)
            g.makeMatingPairs()
            g.setupCycleDay()
            g.day += 1
    return ''

@app.route('/evolve', methods=['POST'])
def evolve():

    global model
    last_generation = model.generation == model.nGenerations
    speed = float(request.form['speed'])
    
    if (model is not None) & (model.generation < model.nGenerations):
            
        # slow speed if close to end
        speed = speed if model.generation + speed < model.nGenerations else model.nGenerations - model.generation

        for gen in np.arange(speed):
            g = 0
            while g < model.nGroups:
                model.groups[g].runModel() 
                g += 1

            # if model.realTimePlots == True and (rd.uniform(0,1) < model.realTimePlotsRate or model.generation == 1):
            #     model.plotSwelling() if model.whichPlot == "swelling" else model.plotPairs()
            # elif rd.uniform(0,1) > 0.99:
            #     print(model.generation)
            
            #model.updateAlphaMatingDays()
            
            # if model.generation == nGenerations - 1:
                # get_ipython().run_line_magic('matplotlib', 'inline')
                # model.plotRS()

            if (model.generation < model.nGenerations):
                # print(model.generation)
                for g in model.groups:
                    g.setupNextGen()
                    g.setGenotypes()
                    g.generation += 1
                    
                if model.dispersal == True:
                    model.migration()

            model.generation += 1

        return str(last_generation)
    else:
        return 'stop'

@app.route('/image-endpoint', methods=['POST'])
def image_endpoint():

    rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})

    global model
    rank = [m.rank for m in model.groups[1].males]
    competitiveness = [m.competitiveness for m in model.groups[1].males]
    reproductiveSuccess = [m.reproductiveSuccess for m in model.groups[1].males]

    plt.close()
    fig, ax = plt.subplots()
    plt.xlabel('Rank', labelpad=15)
    plt.ylabel('Competitive Ability', labelpad=15)
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.82, right=0.95)
    plt.title('Male Rank and Competitive Ability', pad=20, fontweight="bold", fontname="sans-serif", fontsize=20, x = 0.48)
    ax.scatter(rank, competitiveness)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=75)
    buf.seek(0)
    rank_competitiveness = base64.b64encode(buf.read()).decode('ascii')

    # return jsonify(rank_competitiveness=rank_competitiveness, rank_RS=rank_RS)
    return rank_competitiveness

@app.route('/image-endpoint2', methods=['POST'])
def image_endpoint2():

    rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})

    global model

    lst = []
    lstLower = []
    lstUpper =[]
    for j in range(model.cycleLength):
        lst.append(statistics.mean([f.swellingList[j] for f in sum([g.females for g in model.groups], [])]))
        SEM = scipy.stats.tstd([f.swellingList[j] for f in sum([g.females for g in model.groups], [])])
        lstLower.append(lst[j] - SEM) if SEM < lst[j] else lstLower.append(0)
        lstUpper.append(lst[j] + SEM)

    heights = [l + u for l,u in zip(lst,lstUpper)
               ]
    # print("Means:" + str(lst))
    # print("Lower:" +str(lstLower))
    # print("Upper:" +str(lstUpper))
    # print([[f.swellingList[j] for f in sum([g.females for g in model.groups], [])] for j in range(model.cycleLength)])

    plt.close()
    fig, ax = plt.subplots()
    plt.xlabel('Day of the cycle', labelpad=15)
    plt.ylabel('Swelling size', labelpad=13)
    plt.tight_layout(pad=1.8)
    plt.subplots_adjust(top=0.8, right=0.97)
    plt.title('Mean Swelling Size Each Day', pad=20, fontweight="bold", fontname="sans-serif", fontsize=20, x = 0.48)
    ax.plot(lst, "bo")
    [ax.plot([i,i],[l,u], "r") for i,l,u in zip(range(model.cycleLength),lstLower,lstUpper)]
    #plt.hist([i.genes[0] for i in model.groups[1].females])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.ylim(0-(max(lst) / 10) - 0.0001,(max(heights) + 0.001))
    print(max(heights))
    #plt.text(0.1, max(lst) * 0.9, str(self.generation))
    # ax.pause(0.000001)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=75)
    buf.seek(0)
    swelling_plot = base64.b64encode(buf.read()).decode('ascii')

    # rank = [m.rank for m in model.groups[1].males]
    # competitiveness = [m.competitiveness for m in model.groups[1].males]
    # reproductiveSuccess = [m.reproductiveSuccess for m in model.groups[1].males]

    # plt.close()
    # fig, ax = plt.subplots()
    # ax.scatter(rank, reproductiveSuccess, s=15)
    # buf = BytesIO()
    # plt.savefig(buf, format='png', dpi=75)
    # buf.seek(0)
    # rank_RS = base64.b64encode(buf.read()).decode('ascii')

    # return jsonify(rank_competitiveness=rank_competitiveness, rank_RS=rank_RS)
    return swelling_plot

@app.route('/data')
def get_data():
    # Perform necessary operations and retrieve the data
    global model
    rank = [m.rank for m in model.groups[1].males]
    competitiveness = [m.competitiveness for m in model.groups[1].males]
    reproductiveSuccess = [m.reproductiveSuccess for m in model.groups[0].males]
    day = [g.day for g in model.groups]

    # Return the data as JSON
    return "Generation:" + str(model.generation)

@app.route('/execute-command', methods=['POST'])
def execute_command():
    command = request.json['command']
    
    global result 
    # Execute the Python command (using appropriate security measures)
    result = '> ' + str(eval(str(command))) if result == '' else result + '\n> ' + str(eval(str(command)))
    # Return the result as a JSON response
    print(result)
    return json.dumps(result)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8000, debug=True)