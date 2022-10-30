# modulated-DQN
Begleitcode zu meiner Maturarbeit, welches sich um eine modifizierte Version von DQN beschäftigt. 
Der Code wird in zwei Teile unterteilt: 
1. Eine eigene Implementation von DQN und den Modifikationen.
2. Eine Implementation mittels Rllib, welches mit dem Ray Framework arbeitet.

## Die eigene Implementation von (mod.) DQN
Die eigene Implementation ist im wesentlichen nur von Tensorflow und Numpy abhängig. Die finale Version nutzt die Grafikkarte um die Berechnung der temporalen Differenz zu beschleunigen, nutzt aber nur eine Umgebung um Stichproben zu generieren. Die Dateien im Ordner "DQN Implementations" waren ein erster Versuch im Bereich des verstärkten Lernerns. Die aussgesuchten Modifikationen versuchen die Probleme des originalen DQN zu lösen unter der Bedingungung einer einfachen Implementation.




## Anhängigkeiten
Es werden die folgenden Module und Libraries benötigt:
- Python 3 >= 3.8
- tensorflow==2.9.2
- tensorflow-probability==0.17.0
- ray[rllib] >= 2.0.0
- gym[atari]==0.23.1
- lz4==4.0.2
- import-ipynb==0.1.4 (Falls die eigenen Implementationen ausgeführt werden sollen)
