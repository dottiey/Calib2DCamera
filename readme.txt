
INTRINSECI
python calib.py calib_circle -W 7 -H 7 -d 3.75 nomeFileSalvataggioParametri "cerchi\*.png"
OPPURE
python calib.py calib_chessboard -W 10 -H 7 -d 3.75 nomeFileSalvataggioParametri "scacchi\*.png"


ESTRINSECI

python estrinseci.py calib_laser_plane Marco "opencv\\laser\\*.tif" 20 80

example
python calib.py calib_circle -W 7 -H 7 -d 6 Cam1 "cam1\plate\*.tif"
python estrinseci.py calib_laser_plane Cam1 "cam1\laser\*.tif" 23 80
