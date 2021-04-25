from datetime import datetime
import pandas as pd

datos = [1,2,3,4,5,6,7]
fecha = datetime.now().strftime("%m:%d:%Y")
hora = datetime.now().strftime("%H:%M:%S")
hora_vuelo = fecha + '-' + hora
df = pd.DataFrame(datos)
df.to_csv('datos-'+hora_vuelo+'.csv')