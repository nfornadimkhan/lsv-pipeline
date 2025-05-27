# Landessortenversuch (LSV)

State Variety Trial for Germany pipeline

- The aim of this pipline is to extract data from pdf file reports from State Variety Trial for germany. These are in german language.
- Loop one by one on the pdf fils in input folder and extract data to invividual excel files.
- The data is in year wise reports. The year is usullay mentioned on the top of the tables or in the first page with report title.
- If in the table inthe top Wertprüfung* or Wertprüfung, you do not have to include those tables. (you dont have to extract because we have the data already and its actually OVT and not PRT). ignore data from such tables. Like in the below table:

| Sorte                   | Qualität | **Osterseeon** |        |       | **Köfering** |        |        | **Greimersdorf** |       |       | **Giebelstadt** |        |        | **Günzburg** |        |        |
| ----------------------- | --------- | -------------------- | ------ | ----- | ------------------- | ------ | ------ | ---------------------- | ----- | ----- | --------------------- | ------ | ------ | ------------------- | ------ | ------ |
|                         |           | St 1                 | St 2   | MW    | St 1                | St 2   | MW     | St 1                   | St 2  | MW    | St 1                  | St 2   | MW     | St 1                | St 2   | MW     |
| **Wertprüfung*** |           |                      |        |       |                     |        |        |                        |       |       |                       |        |        |                     |        |        |
| LG Initial              | A         | 91,84                | 97,61  | 94,73 | 106,64              | 110,49 | 108,57 | 75,25                  | 78,69 | 76,97 | 102,50                | 109,04 | 105,77 | 96,90               | 107,52 | 102,21 |
| ASUR 06587              |           | 97,72                | 101,64 | 99,68 | 115,39              | 122,56 | 118,97 | 84,73                  | 88,91 | 86,82 | 112,00                | 115,99 | 114,00 | 104,08              | 114,74 | 109,41 |
| NORD 06592              |           | 89,27                | 97,70  | 93,49 | 104,51              | 104,63 | 104,57 | 83,13                  | 80,20 | 81,67 | 108,04                | 110,11 | 109,08 | 100,62              | 105,94 | 103,28 |
| SECO 06609              |           | 93,53                | 96,38  | 94,96 | 110,67              | 118,61 | 114,64 | 78,89                  | 81,90 | 80,40 | 104,78                | 108,20 | 106,49 | 103,64              | 111,46 | 107,55 |

- Only want to extract data related to "Winterweizen".
- Only want to extract "Kornertrag absolut" (absolute yield which is 'Kornertrag absolut").
- There are some tables like this:

Kornertrag absolut, Sorten, Orte und Behandlungen, 2023

| Sorte                                   | Qualität | **Osterseeon** |        |        | **Köfering** |        |        | **Greimersdorf** |       |       | **Giebelstadt** |        |        | **Günzburg** |        |        |
| --------------------------------------- | --------- | -------------------- | ------ | ------ | ------------------- | ------ | ------ | ---------------------- | ----- | ----- | --------------------- | ------ | ------ | ------------------- | ------ | ------ |
|                                         |           | St 1                 | St 2   | MW     | St 1                | St 2   | MW     | St 1                   | St 2  | MW    | St 1                  | St 2   | MW     | St 1                | St 2   | MW     |
| **LSV Hauptsortiment**            |           |                      |        |        |                     |        |        |                        |       |       |                       |        |        |                     |        |        |
| Axioma                                  | E         | 75,55                | 85,80  | 80,67  | 97,33               | 106,08 | 101,70 | 70,10                  | 74,34 | 72,22 | 94,95                 | 102,99 | 98,97  | 79,77               | 91,93  | 85,85  |
| KWS Emerick                             | E         | 92,82                | 101,68 | 97,25  | 106,92              | 111,55 | 109,24 | 83,41                  | 82,25 | 82,83 | 98,96                 | 106,28 | 102,62 | 96,15               | 104,27 | 100,21 |
| Viki                                    | E         | 85,30                | 89,16  | 87,23  | 99,76               | 106,23 | 102,99 | 66,78                  | 71,44 | 69,11 | 85,34                 | 94,31  | 89,83  | 84,99               | 97,43  | 91,21  |
| Exsal                                   | E         | 87,24                | 90,53  | 88,89  | 103,92              | 113,05 | 108,49 | 87,22                  | 82,09 | 84,66 | 106,99                | 104,42 | 105,70 | 101,52              | 105,57 | 103,55 |
| Patras                                  | A         | 86,99                | 97,71  | 92,35  | 101,31              | 109,92 | 105,61 | 76,38                  | 86,12 | 81,25 | 101,58                | 105,86 | 103,72 | 90,78               | 103,23 | 97,00  |
| RGT Reform                              | A         | 90,66                | 98,56  | 94,61  | 103,04              | 113,67 | 108,35 | 76,61                  | 72,91 | 74,76 | 99,52                 | 104,94 | 102,23 | 87,22               | 96,91  | 92,07  |
| Apostel                                 | A         | 90,68                | 94,18  | 92,43  | 106,56              | 115,32 | 110,94 | 76,86                  | 85,67 | 81,31 | 98,47                 | 105,92 | 102,20 | 90,21               | 103,76 | 96,99  |
| Asory                                   | A         | 88,08                | 100,45 | 94,26  | 112,58              | 118,51 | 115,55 | 77,43                  | 83,13 | 80,28 | 103,70                | 109,63 | 106,66 | 100,97              | 110,01 | 105,49 |
| Foxx                                    | A         | 89,45                | 95,80  | 92,63  | 96,02               | 104,25 | 100,25 | 79,68                  | 78,75 | 79,20 | 101,71                | 102,79 | 102,25 | 95,05               | 106,30 | 100,68 |
| Akzent                                  | A         | 90,46                | 93,97  | 92,22  | 98,47               | 106,27 | 102,37 | 89,11                  | 83,90 | 86,51 | 97,03                 | 109,22 | 103,17 | 93,03               | 105,07 | 99,05  |
| Hyvega                                  | A         | 94,11                | 100,33 | 97,22  | 111,82              | 108,06 | 109,94 | 86,27                  | 83,74 | 85,01 | 109,90                | 103,46 | 106,48 | 109,61              | 109,46 | 105,19 |
| LG Character                            | A         | 93,35                | 93,67  | 93,51  | 108,14              | 108,49 | 108,31 | 77,47                  | 84,77 | 81,12 | 98,34                 | 109,06 | 103,70 | 96,05               | 107,02 | 101,53 |
| KWS Donovan                             | A         | 90,22                | 99,80  | 95,01  | 107,11              | 107,14 | 107,12 | 83,66                  | 95,02 | 89,34 | 94,56                 | 114,55 | 104,55 | 98,16               | 110,34 | 104,25 |
| SU Jonte                                | A         | 87,65                | 95,12  | 91,39  | 108,78              | 110,88 | 109,83 | 90,66                  | 83,84 | 87,24 | 102,32                | 108,74 | 105,53 | 94,09               | 100,87 | 97,48  |
| LG Atelier                              | A         | 89,14                | 94,41  | 91,78  | 106,53              | 112,10 | 109,32 | 78,69                  | 79,13 | 78,91 | 95,11                 | 105,54 | 100,33 | 98,20               | 108,66 | 103,43 |
| Absolut                                 | A         | 85,60                | 89,92  | 87,76  | 104,58              | 109,71 | 107,15 | 72,40                  | 79,39 | 75,90 | 101,09                | 106,90 | 104,00 | 94,48               | 103,73 | 99,11  |
| Polarkap                                | A         | 95,57                | 97,19  | 96,38  | 108,56              | 115,17 | 111,86 | 80,53                  | 79,05 | 79,79 | 105,15                | 108,42 | 106,78 | 102,06              | 109,87 | 105,97 |
| Cayenne                                 | A         | 87,07                | 97,67  | 92,37  | 102,18              | 111,83 | 107,00 | 75,35                  | 77,51 | 76,43 | 103,65                | 105,25 | 104,45 | 99,35               | 101,12 | 100,24 |
| Absint                                  | A         | 91,53                | 100,09 | 95,81  | 104,52              | 115,79 | 110,16 | 74,51                  | 79,60 | 77,05 | 100,82                | 105,35 | 103,09 | 100,37              | 105,55 | 102,96 |
| Adrenalin                               | A         | 91,56                | 102,75 | 97,16  | 111,34              | 120,51 | 115,92 | 80,72                  | 82,22 | 81,47 | 105,23                | 113,74 | 109,49 | 106,53              | 111,90 | 109,21 |
| LG Optimist                             | A         | 93,10                | 110,65 | 105,55 | 114,25              | 123,26 | 118,85 | 91,33                  | 85,70 | 88,51 | 111,29                | 110,13 | 110,71 | 117,90              | 114,01 | 116,01 |
| RGT Kreation                            | A         | 98,12                | 102,40 | 100,26 | 107,43              | 116,87 | 112,15 | 83,45                  | 88,07 | 85,76 | 107,01                | 111,85 | 109,43 | 104,02              | 108,30 | 106,16 |
| **Mittel dt/ha (Hauptsortiment)** |           | 90,91                | 97,84  | 94,38  | 106,11              | 112,91 | 109,51 | 79,03                  | 83,96 | 81,49 | 101,85                | 109,02 | 105,43 | 98,42               | 107,39 | 102,91 |

From the above such tables i want to extract the above data in this format:
in data/output folder with file name LSV_de_WW_2023. (year from the table) in this file:

Year	Trial	            Trait	                                                Variety	    Location	Treatment       Value
2023	2023_whw_de_prt_lsv	Kornertrag absolut, Sorten, Orte und Behandlungen, 2023	Axioma  	Osterseeon	St 1	        75.55
2023	2023_whw_de_prt_lsv	Kornertrag absolut, Sorten, Orte und Behandlungen, 2023	Axioma 	    Osterseeon	St 2            85.80
2023	2023_whw_de_prt_lsv	Kornertrag absolut, Sorten, Orte und Behandlungen, 2023	Axioma	    Köfering	St 1	        97.33
2023	2023_whw_de_prt_lsv	Kornertrag absolut, Sorten, Orte und Behandlungen, 2023	Axioma	    Köfering	St 2	        106.08
.... like this for all.

So you have to ignore the MW which are mean values. I am not interested in the mean values. Only absolute values.
Similarly the end coloumn if contrains mean values like in the above ("Mittel dt/ha")
If value is empty then use na.

The there are some table with this heading: "Kornertrag relativ, Sorten, Orte und Behandlungen, 2023 - Fortsetzung"
ignore such tables are these contains relative values: "relativ" means this table contrains relative values.

Some table say St 1 as Stufe 1 and St 2 as Stufe 2 and MW to mittel.

- Two creteria for extraction.

1. If the values are absolute and mean then take only absolute values for yield trait for a single file.
2. If values are only relative then take the relative values and find the standard from which this are converted to relative and then convert these to absolute values for a single file.

Then there are files like this:

Erträge
Erträge, Absoluter Ertrag: Korn (dt/ha, 86 % TS), unbehandelte Stufe

| Sorte                | Qual.-Gr. | **Lössböden** |                    |                    |                    | **Verwitterungsböden** |                    |                    |                    |
| -------------------- | --------- | --------------------- | ------------------ | ------------------ | ------------------ | ----------------------------- | ------------------ | ------------------ | ------------------ |
|                      |           | (B)                   | 2021 `<br>`n = 9 | 2022 `<br>`n = 9 | 2023 `<br>`n = 8 | (B)                           | 2021 `<br>`n = 5 | 2022 `<br>`n = 3 | 2023 `<br>`n = 5 |
| KWS Emerick          | E         | x                     | 82,4               | 91,6               | 95,1               | x                             | 76,0               | 82,0               | 88,5               |
| Moschus              | E         | x                     | 80,2               | 87,3               | 91,9               | x                             | 75,4               | 76,2               | 81,8               |
| Ponticus             | E         | x                     | 82,9               | 89,1               | 95,3               |                               |                    |                    |                    |
| Exsal                | E         |                       |                    |                    | 98,6               |                               |                    |                    | 90,4               |
| Asory                | A         | x                     | 82,9               | 90,8               | 94,5               | x                             | 81,8               | 83,6               | 89,2               |
| Attribut             | A         | x                     | 87,5               | 93,6               | 96,4               | x                             | 82,3               | 75,6               | 93,5               |
| Foxx                 | A         | x                     | 85,0               | 92,8               | 97,2               | x                             | 82,6               | 78,4               | 89,1               |
| Hyvega (H)           | A         |                       |                    |                    |                    | x                             | 86,5               | 83,0               | 94,0               |
| KWS Donovan          | A         | x                     | 93,4               | 93,9               | 90,2               | x                             | 85,3               | 84,5               | 79,9               |
| KWS Imperium         | A         | x                     | 85,8               | 96,8               | 99,0               |                               |                    |                    |                    |
| Lemmy                | A         | x                     | 84,2               | 92,3               | 96,0               | x                             | 76,3               | 76,6               | 86,2               |
| LG Character         | A         |                       |                    |                    |                    | x                             | 80,3               | 82,6               |                    |
| LG Initial           | A         | x                     | 86,4               | 93,4               | 95,3               | x                             | 77,9               |                    | 87,8               |
| Patras               | A         | x                     | 84,5               | 90,4               | 93,1               |                               |                    |                    |                    |
| RGT Depot            | A         | x                     | 85,0               | 96,4               | 97,5               | x                             | 76,7               | 79,4               | 91,2               |
| RGT Reform           | A         | x                     | 81,1               | 91,9               | 99,3               | x                             | 76,3               | 77,5               | 93,0               |
| SU Jonte             | A         | x                     | 88,2               | 91,5               | 99,4               | x                             | 81,4               | 75,9               | 90,2               |
| Absolut              | A         |                       |                    |                    | 95,9               |                               |                    |                    |                    |
| KWS Mitchum          | A         |                       |                    |                    | 93,0               |                               |                    |                    |                    |
| LG Atelier           | A         |                       | 93,1               | 94,1               | 94,9               |                               |                    |                    | 90,4               |
| Polarkap             | A         |                       | 89,4               |                    | 96,2               |                               | 75,2               |                    | 90,7               |
| SU Willem            | A         |                       |                    |                    | 93,6               |                               |                    |                    | 86,9               |
| Absint               | A         |                       |                    |                    |                    |                               |                    |                    | 89,2               |
| Cayenne              | A         |                       |                    |                    | 91,2               |                               |                    |                    | 88,0               |
| Adrenalin            | A         |                       |                    |                    | 96,8               |                               |                    |                    |                    |
| LG Optimist          | A         |                       |                    |                    | 101,7              |                               |                    |                    | 100,6              |
| RGT Kreation         | A         |                       |                    |                    | 100,0              |                               |                    |                    | 95,2               |
| Campesino            | B         |                       |                    |                    |                    | x                             | 84,4               | 86,8               | 101,6              |
| Chevignon            | (B)       | x                     | 90,1               | 97,8               | 103,4              | x                             | 84,9               | 95,5               | 98,8               |
| Complice             | (B)       | x                     | 89,7               | 99,2               | 104,9              |                               |                    |                    |                    |
| Informer             | B         | x                     | 86,1               | 95,3               | 94,4               | x                             | 79,2               | 81,9               | 89,2               |
| Knut                 | B         |                       |                    |                    |                    | x                             | 83,5               | 84,2               | 100,2              |
| Debian               | B         |                       | 93,5               |                    | 99,1               |                               | 85,0               | 88,3               |                    |
| KWS Mintum           | B         |                       |                    |                    |                    |                               |                    |                    | 93,5               |
| Spectral             | B         |                       |                    |                    |                    |                               |                    |                    | 95,5               |
| **Mittel (B)** |           |                       | **85,6**     | **93,0**     | **96,3**     |                               | **80,8**     | **80,6**     | **91,0**     |

Sortenversuch Frühsaat Winterweizen 2023: early sowing do not take this values
Sortenversuch Spätsaat Winterweizen 2023: late sowen do not take these alos   Spätsaatversuchen NRW 2023

# How to run script

```bash
python main.py lsv 
```
