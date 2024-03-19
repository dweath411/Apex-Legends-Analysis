#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:42:55 2024

@author: derienweatherspoon
"""

# %% Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Read Data

# read in file
file_path = "/Users/derienweatherspoon/Desktop/a/apex2022.csv"

# read in as pandas dataframe
data = pd.read_csv(file_path)   

# drop columns for simplicity
dropped_columns = ['Time of Day', 'Week']  

# new data with dropped columns
data.drop(columns=dropped_columns, inplace = True)  

# lower text
data.columns = data.columns.str.lower().str.replace(' ', '_') 

# preview new data
print(data.head())

# %% Checking data

# fix wrong capitalization
data['dlegend'].replace({'GIbraltar': 'Gibraltar'}, inplace=False)

# count null values
print(data.isnull().sum())  

# describe data 
described_data = data.describe()
print(described_data)

# %% Usage per legend, per user

# initialize 
legend_counts = {}

# list of columns containing legend data
legend_columns = ['dlegend', 'plegend', 'slegend']

# iterate over each column
for column in legend_columns:
    # count the total number of times each user has used each legend
    counts = data[column].value_counts().reset_index()
    # store the counts in the dictionary with column name as key
    legend_counts[column] = counts

# counts for each column
for column, counts in legend_counts.items():
    print(column, 'counts:')
    print(counts)

    
# %% Highest damage per user, per legend

# initialize 
highest_damage = {}

# list of columns containing damage data
damage_columns = ['djdmg', 'popdmg', 'spoondmg']

# list of corresponding legend column names
legend_columns = ['dlegend', 'plegend', 'slegend']

# iterate over each pair of damage and legend columns
for damage_column, legend_column in zip(damage_columns, legend_columns):
    # get the highest damage for each user per character
    highest_damage[damage_column] = data.groupby(data[legend_column])[damage_column].max().sort_values(ascending=False).reset_index()

# print
for damage_column, highest_damage_df in highest_damage.items():
    print(highest_damage_df)
    
# %% Highest kills per user, per legend

# initialize 
highest_kills = {}

# list of columns containing damage data
kill_columns = ['djkill', 'popkill', 'spoonkill']

# list of corresponding legend column names
legend_columns = ['dlegend', 'plegend', 'slegend']

# iterate over each pair of kills and legend columns
for kill_column, legend_column in zip(kill_columns, legend_columns):
    # get the highest kills for each user per character
    highest_kills[kill_column] = data.groupby(data[legend_column])[kill_column].max().sort_values(ascending=False).reset_index()

# print
for kill_column, highest_kill_df in highest_kills.items():
    print(highest_kill_df)
    
# %% Correlation matrix

# I wanted to see if anyone else's stats played a factor in another persons.

# select only the numeric columns for the correlation matrix
numeric_data = data[['djdmg', 'djkill', 'popdmg', 'popkill', 'spoondmg', 'spoonkill']]

# calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# print
print(correlation_matrix)

# %% Summation data per user

# initialize 
legend_summation = {}

# list of columns and corresponding legend columns
columns = ['djdmg', 'djkill', 'popdmg', 'popkill', 'spoondmg', 'spoonkill']
legend_columns = ['dlegend', 'plegend', 'slegend']

# iterate over each pair of columns and legend columns
for damage_column, legend_column in zip(columns[::2], legend_columns):
    # group by legend and aggregate sum of damage and kills
    legend_summation[legend_column] = data.groupby(legend_column).agg({damage_column: 'sum', columns[columns.index(damage_column) + 1]: 'sum'}).sort_values(by=[damage_column, columns[columns.index(damage_column) + 1]], ascending=False).reset_index()

# print
for legend_column, legend_df in legend_summation.items():
    print(f"{legend_column} summation:")
    print(legend_df)
    
    
# %% Best legend for each user

# best legend for each user (most damage and kills)

best_legend_dj = data.groupby('dlegend').agg({'djdmg': 'max', 'djkill': 'max'}).sort_values(by=['djdmg', 'djkill'], ascending=False).reset_index()
best_legend_pop = data.groupby('plegend').agg({'popdmg': 'max', 'popkill': 'max'}).sort_values(by=['popdmg', 'popkill'], ascending=False).reset_index()
best_legend_spoon = data.groupby('slegend').agg({'spoondmg': 'max', 'spoonkill': 'max'}).sort_values(by=['spoondmg', 'spoonkill'], ascending=False).reset_index()

print(best_legend_dj)
print(best_legend_pop)
print(best_legend_spoon)

# %% Initialize summation stats for plots for each user

# sum the damage done by each user for each legend
dj_damage_per_legend = data.groupby('dlegend')['djdmg'].sum().reset_index()
pop_damage_per_legend = data.groupby('plegend')['popdmg'].sum().reset_index()
spoon_damage_per_legend = data.groupby('slegend')['spoondmg'].sum().reset_index()

# %% Summation plots for DJ

# damage distribution per legend for dj

# pie chart
plt.figure(figsize=(8, 8))
plt.pie(dj_damage_per_legend['djdmg'], labels=dj_damage_per_legend['dlegend'],autopct='%1.1f%%', labeldistance=1.3)
plt.title('DJ Damage Distribution by Legend')
plt.show()

# horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(dj_damage_per_legend['dlegend'], dj_damage_per_legend['djdmg'], color='lightgreen')
plt.xlabel('Total Damage')
plt.ylabel('Legend')
plt.title('Total DJ Damage by Legend')
plt.tight_layout()
plt.show()

# %% Summation plots for Pop

# damage distribution per legend for Pop
explode_p = [0.1] * len(pop_damage_per_legend)

# pie chart
plt.figure(figsize=(8, 8))
plt.pie(pop_damage_per_legend['popdmg'], labels=pop_damage_per_legend['plegend'], explode = explode_p, autopct='%1.1f%%', startangle=140, labeldistance=1.3)
plt.title('Pop Damage Distribution by Legend')
plt.show()

# horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(pop_damage_per_legend['plegend'], pop_damage_per_legend['popdmg'], color='red')
plt.xlabel('Total Damage')
plt.ylabel('Legend')
plt.title('Total Pop Damage by Legend')
plt.tight_layout()
plt.show()
# %% Summation plots for Spoon

# damage distribution per legend for Spoon
explode_s = [0.1] * len(spoon_damage_per_legend)

# Pie chart
# plt.figure(figsize=(8, 8))
# plt.pie(spoon_damage_per_legend['spoondmg'], labels=spoon_damage_per_legend['slegend'], explode = explode_s, autopct='%1.1f%%', startangle=140, labeldistance=1.3)
# plt.title('Spoon Damage Distribution by Legend')
# plt.show()

# Spoon pie chart archived due to character distribution extremities.

# horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(spoon_damage_per_legend['slegend'], spoon_damage_per_legend['spoondmg'], color='lightblue')
plt.xlabel('Total Damage')
plt.ylabel('Legend')
plt.title('Total Spoon Damage by Legend')
plt.tight_layout()
plt.show()

# %% Max damage plots for DJ

# get best legend variables set for dj
dj_legends = best_legend_dj['dlegend']
dj_max_damage = best_legend_dj['djdmg']
dj_max_kills = best_legend_dj['djkill']

# plot max damage
plt.figure(figsize=(10, 5))
plt.bar(dj_legends, dj_max_damage, color='green')
plt.title('Highest Damage per Legend (DJ)')
plt.xlabel('Legend')
plt.ylabel('Damage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot max kills
plt.figure(figsize=(10, 5))
plt.bar(dj_legends, dj_max_kills, color='darkgreen')
plt.title('Highest Kills per Legend (DJ)')
plt.xlabel('Legend')
plt.ylabel('Kills')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# NOT on R file
# %% Max damage plots for spoon

# get best legend variables set for spoon
spoon_legends = best_legend_spoon['slegend']
spoon_max_damage = best_legend_spoon['spoondmg']
spoon_max_kills = best_legend_spoon['spoonkill']

# plot max damage
plt.figure(figsize=(10, 5))
plt.bar(spoon_legends, spoon_max_damage, color='blue')
plt.title('Highest Damage per Legend (Spoon)')
plt.xlabel('Legend')
plt.ylabel('Damage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot max kills
plt.figure(figsize=(10, 5))
plt.bar(spoon_legends, spoon_max_kills, color='blue')
plt.title('Highest Kills per Legend (Spoon)')
plt.xlabel('Legend')
plt.ylabel('Kills')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# NOT on R file

# %% Getting highest damage and kills as variables

# do not print these, only used to combine data for plotting

highest_damage_dj = data.groupby('dlegend')['djdmg'].max().sort_values(ascending=False).reset_index()
highest_damage_pop = data.groupby('plegend')['popdmg'].max().sort_values(ascending=False).reset_index()
highest_damage_spoon = data.groupby('slegend')['spoondmg'].max().sort_values(ascending=False).reset_index()
highest_kills_dj = data.groupby('dlegend')['djkill'].max().sort_values(ascending=False).reset_index()
highest_kills_pop = data.groupby('plegend')['popkill'].max().sort_values(ascending=False).reset_index()
highest_kills_spoon = data.groupby('slegend')['spoonkill'].max().sort_values(ascending=False).reset_index()
dj_summation = data.groupby('dlegend').agg({'djdmg': 'sum', 'djkill': 'sum'}).sort_values(by=['djdmg', 'djkill'], ascending=False).reset_index()
pop_summation = data.groupby('plegend').agg({'popdmg': 'sum', 'popkill': 'sum'}).sort_values(by=['popdmg', 'popkill'], ascending=False).reset_index()
spoon_summation = data.groupby('slegend').agg({'spoondmg': 'sum', 'spoonkill': 'sum'}).sort_values(by=['spoondmg', 'spoonkill'], ascending=False).reset_index()
# %% Combined damage and kills 

# combine the data for plotting
combined_damage_max = pd.concat([highest_damage_dj.rename(columns={'dlegend': 'legend', 'djdmg': 'damage'}),
                             highest_damage_pop.rename(columns={'plegend': 'legend', 'popdmg': 'damage'}),
                             highest_damage_spoon.rename(columns={'slegend': 'legend', 'spoondmg': 'damage'})])

combined_kills_max = pd.concat([highest_kills_dj.rename(columns={'dlegend': 'legend', 'djkill': 'kills'}),
                            highest_kills_pop.rename(columns={'plegend': 'legend', 'popkill': 'kills'}),
                            highest_kills_spoon.rename(columns={'slegend': 'legend', 'spoonkill': 'kills'})])

combined_damage_sum = pd.concat([dj_summation.rename(columns={'dlegend': 'legend', 'djdmg': 'damage'}),
                             pop_summation.rename(columns={'plegend': 'legend', 'popdmg': 'damage'}),
                             spoon_summation.rename(columns={'slegend': 'legend', 'spoondmg': 'damage'})])

combined_kills_sum = pd.concat([dj_summation.rename(columns={'dlegend': 'legend', 'djkill': 'kills'}),
                            pop_summation.rename(columns={'plegend': 'legend', 'popkill': 'kills'}),
                            spoon_summation.rename(columns={'slegend': 'legend', 'spoonkill': 'kills'})])

# NOT on R file

# %% Plots for combined data

# extract data for plotting
legends_damage_max = combined_damage_max['legend']
damage_values_max = combined_damage_max['damage']

legends_damage_sum = combined_damage_sum['legend']
damage_values_sum = combined_damage_sum['damage']

legends_kills_max = combined_kills_max['legend']
kills_values_max = combined_kills_max['kills']

legends_kills_sum = combined_kills_sum['legend']
kills_values_sum = combined_kills_sum['kills']


# plot damage (MAX)
plt.figure(figsize=(10, 5))
plt.bar(legends_damage_max, damage_values_max, color='blue')
plt.title('Highest Damage by Legend (For all 3 users)')
plt.xlabel('Legend')
plt.ylabel('Damage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot kills (MAX)
plt.figure(figsize=(10, 5))
plt.bar(legends_kills_max, kills_values_max, color='green')
plt.title('Highest Kills by Legend (For all 3 users)')
plt.xlabel('Legend')
plt.ylabel('Kills')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot damage (SUM)
plt.figure(figsize=(10, 5))
plt.bar(legends_damage_sum, damage_values_sum, color='purple')
plt.title('Damage summations by Legend (For all 3 users)')
plt.xlabel('Legend')
plt.ylabel('Damage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot kills (SUM)
plt.figure(figsize=(10, 5))
plt.bar(legends_kills_sum, kills_values_sum, color='red')
plt.title('Kill summations by Legend (For all 3 users)')
plt.xlabel('Legend')
plt.ylabel('Kills')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Extreme outlier of Octane, since Spoon has over 325 games as Octane.

# NOT on R file

# %% Individual 1st place frequencies for DJ

# filter for 1st place placings for DJ
first_place_dj_data = data[(data['placing'] == 1)]

# group by DJ's legend and location to get the frequency of 1st place finishes
frequency_dj_data = first_place_dj_data.groupby(['dlegend', 'location']).size().reset_index(name='frequency')

# get unique legends and locations
unique_legends_dj = frequency_dj_data['dlegend'].unique()
unique_locations = data['location'].unique()

# number of bars for each legend
n_bars = len(unique_locations)

# set up the matplotlib figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# bar width
bar_width = 0.8 / n_bars

# set the positions of the bars
indices = np.arange(len(unique_legends_dj))

# plotting each location's frequency per legend
for i, location in enumerate(unique_locations):
    location_data = frequency_dj_data[frequency_dj_data['location'] == location]
    # ensure the order of legends is consistent
    ordered_frequencies = location_data.set_index('dlegend').reindex(unique_legends_dj).fillna(0)
    ax.bar(indices + i * bar_width, ordered_frequencies['frequency'], width=bar_width, label=location)

# set y-range to 25
ax.set_ylim(0, 25) 

# make the y-axis go in increments of 3
ax.set_yticks(np.arange(0, 26, 3))

# add labels and title
ax.set_xlabel('DJ Legend')
ax.set_ylabel('Frequency of 1st Place')
ax.set_title('Frequency of 1st Place for DJ by Legend and Map')
ax.set_xticks(indices + bar_width * (n_bars - 1) / 2)
ax.set_xticklabels(unique_legends_dj)
ax.legend(title='Map')

# rotate x-axis labels
plt.xticks(rotation=90)

# show the plot
plt.tight_layout()
plt.show()

# %% Individual 1st place frequencies for Pop

# filter for 1st place placings for pop
first_place_pop_data = data[(data['placing'] == 1)]

# group by pop's legend and location to get the frequency of 1st place finishes
frequency_pop_data = first_place_pop_data.groupby(['plegend', 'location']).size().reset_index(name='frequency')

# get unique legends and locations
unique_legends_pop = frequency_pop_data['plegend'].unique()
unique_locations = data['location'].unique()

# number of bars for each legend
n_bars = len(unique_locations)

# set up the matplotlib figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# bar width
bar_width = 0.8 / n_bars

# set the positions of the bars
indices = np.arange(len(unique_legends_pop))

# plotting each location's frequency per legend
for i, location in enumerate(unique_locations):
    location_data = frequency_pop_data[frequency_pop_data['location'] == location]
    # ensure the order of legends is consistent
    ordered_frequencies = location_data.set_index('plegend').reindex(unique_legends_pop).fillna(0)
    ax.bar(indices + i * bar_width, ordered_frequencies['frequency'], width=bar_width, label=location)

# set y-range to 27
ax.set_ylim(0, 27) 

# make the y-axis go in increments of 3
ax.set_yticks(np.arange(0, 28, 3))

# add labels and title
ax.set_xlabel('pop Legend')
ax.set_ylabel('Frequency of 1st Place')
ax.set_title('Frequency of 1st Place for Pop by Legend and Map')
ax.set_xticks(indices + bar_width * (n_bars - 1) / 2)
ax.set_xticklabels(unique_legends_pop)
ax.legend(title='Map')

# rotate x-axis labels
plt.xticks(rotation=90)

# show the plot
plt.tight_layout()
plt.show()

# %% Individual 1st place frequencies for Spoon

# filter for 1st place placings for Spoon
first_place_spoon_data = data[(data['placing'] == 1)]

# group by spoon's legend and location to get the frequency of 1st place finishes
frequency_spoon_data = first_place_spoon_data.groupby(['slegend', 'location']).size().reset_index(name='frequency')

# get unique legends and locations
unique_legends_spoon = frequency_spoon_data['slegend'].unique()
unique_locations = data['location'].unique()

# number of bars for each legend
n_bars = len(unique_locations)

# set up the matplotlib figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# bar width
bar_width = 0.8 / n_bars

# set the positions of the bars
indices = np.arange(len(unique_legends_spoon))

# plotting each location's frequency per legend
for i, location in enumerate(unique_locations):
    location_data = frequency_spoon_data[frequency_spoon_data['location'] == location]
    # ensure the order of legends is consistent
    ordered_frequencies = location_data.set_index('slegend').reindex(unique_legends_spoon).fillna(0)
    ax.bar(indices + i * bar_width, ordered_frequencies['frequency'], width=bar_width, label=location)

# set y-range to 40
ax.set_ylim(0, 40) 

# make the y-axis go in increments of 3
ax.set_yticks(np.arange(0, 41, 3))

# add labels and title
ax.set_xlabel('Spoon Legend')
ax.set_ylabel('Frequency of 1st Place')
ax.set_title('Frequency of 1st Place for Spoon by Legend and Map')
ax.set_xticks(indices + bar_width * (n_bars - 1) / 2)
ax.set_xticklabels(unique_legends_spoon)
ax.legend(title='Map')

# rotate x-axis labels
plt.xticks(rotation=90)

# show the plot
plt.tight_layout()
plt.show()

# %% First Place Frequencies, grouped by Legend usage and map for all 3 users

# filter for 1st place placings
first_place_data = data[data['placing'] == 1]

# group by characters and location to get the frequency of 1st place finishes
frequency_data = first_place_data.groupby(['dlegend', 'plegend', 'slegend', 'location']).size().reset_index(name='frequency')

# sort the results
sorted_frequency_data = frequency_data.sort_values(by='frequency', ascending=False)

# display the sorted frequency data
# print(sorted_frequency_data) # not showing for now

# bar plot
fig, ax = plt.subplots(figsize=(10, 8))
# get unique locations to create subplots
locations = frequency_data['location'].unique()
num_locations = len(locations)

# create subplots for each location
for i, location in enumerate(locations):
    location_data = frequency_data[frequency_data['location'] == location]
    ax.bar(location_data['dlegend'] + '_' + location_data['plegend'] + '_' + location_data['slegend'], location_data['frequency'], label=location)

# add labels and title
ax.set_xlabel('Character Combinations')
ax.set_ylabel('Frequency of 1st Place')
ax.set_title('Frequency of 1st Place by Character Lineup and Location')
ax.legend(title='Location')

# rotate x-axis labels
plt.xticks(rotation=90)

# plot
plt.tight_layout()
plt.show()

# %% Are we better as a group in ranked or pubs?

# filter for 1st place placings and mode is either 'pub' or 'ranked'
first_place_pub_data = data[(data['placing'] == 1) & (data['mode'].str.strip().str.lower() == 'pub')]
first_place_ranked_data = data[(data['placing'] == 1) & (data['mode'].str.strip().str.lower() == 'ranked')]

# group by characters and location to get the frequency of 1st place finishes for pub and ranked
frequency_pub_data = first_place_pub_data.groupby(['dlegend', 'plegend', 'slegend', 'location']).size().reset_index(name='frequency')
frequency_ranked_data = first_place_ranked_data.groupby(['dlegend', 'plegend', 'slegend', 'location']).size().reset_index(name='frequency')

# bar plot for PUB
fig_pub, ax_pub = plt.subplots(figsize=(10, 8))
locations = frequency_pub_data['location'].unique()
for i, location in enumerate(locations):
    location_data = frequency_pub_data[frequency_pub_data['location'] == location]
    ax_pub.bar(location_data['dlegend'] + '_' + location_data['plegend'] + '_' + location_data['slegend'], location_data['frequency'], label=location)

# add labels and title for PUB plot
ax_pub.set_xlabel('Character Combinations')
ax_pub.set_ylabel('Frequency of 1st Place')
ax_pub.set_title('Frequency of 1st Place by Character Lineup and Location for PUB Mode')
ax_pub.legend(title='Location')

# rotate x-axis labels for PUB plot
plt.sca(ax_pub)
plt.xticks(rotation=90)

# bar plot for Ranked
fig_ranked, ax_ranked = plt.subplots(figsize=(10, 8))
for i, location in enumerate(locations):
    location_data = frequency_ranked_data[frequency_ranked_data['location'] == location]
    ax_ranked.bar(location_data['dlegend'] + '_' + location_data['plegend'] + '_' + location_data['slegend'], location_data['frequency'], label=location)

# add labels and title for Ranked plot
ax_ranked.set_xlabel('Character Combinations')
ax_ranked.set_ylabel('Frequency of 1st Place')
ax_ranked.set_title('Frequency of 1st Place by Character Lineup and Location for Ranked Mode')
ax_ranked.legend(title='Location')

# rotate x-axis labels for Ranked plot
plt.sca(ax_ranked)
plt.xticks(rotation=90)

# show plots
plt.tight_layout()
plt.show()
# %% Scoring method

# define kill and damage weights
kill_weight = 0.55
damage_weight = 0.45

# define a dict to store scores for each legend
legend_scores = {}

# list of legend columns and their corresponding kill and damage columns
legend_columns = ['dlegend', 'plegend', 'slegend']
kill_columns = ['djkill', 'popkill', 'spoonkill']
damage_columns = ['djdmg', 'popdmg', 'spoondmg']

# iterate over each legend
for legend_column, kill_column, damage_column in zip(legend_columns, kill_columns, damage_columns):
    # compute the score for the current legend
    legend_score = (data[kill_column] * kill_weight) + (data[damage_column] * damage_weight)
    # store the mean of scores divided by 1000 in the dictionary
    legend_scores[legend_column] = legend_score.mean() / 1000

# print
for legend, score in legend_scores.items():
    print(f"Mean score for {legend}: {score}")
    
    