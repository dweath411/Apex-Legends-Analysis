# Apex-Legends-Analysis
An Apex Legends data analysis.
The data set used for this analysis is based on the game Apex Legends. I collected the data myself, over the course of 30+ weeks using the stats provided by the game at the end of every match. The same three players are the only three players in the data. Only games where we placed top 5 are included. Since I collected it myself, there should be no need for any pre-processing. The goal is get an intermediate analysis from our games to see what works best and what may not work the best.

Evaluation may be used using regression models or tree based methods.

Explanation of the variables in the dataset:

djdmg, popdmg, spoondmg: The amount of damage each user accumulated during the match
djkill, popkill, spoonkill: The amount of kills each user accumulated during the match
location: The map the match was played on. There are three maps used in this rotation; World’s Edge (WE), Olympus (OL), and Kings Canyon (KC)
dlegend, plegend, slegend: The legend each user used during that match
placing: The placement we received at the end of the match. This number is 1-5
date: Calendar date the match was played on
day: Calendar day the match was played on
mode: The game mode the match was played on. This is either ranked or pub (public)
