# Librerias empleadas

import cv2
import matplotlib.pyplot as plt
import numpy as np
from djitellopy import tello
import time
import threading
import pandas as pd

# Variables globales

tiempo_flag = False # Se inicializa un tiempo de vuelo
alt_des = 60        # Altura deseada
alt_des1 = 60
vel_des = 0         # Velocidad deseada 

# Función para generar la trayectoria
def gen_trayectoria(T, alt_ini, alt_fin):
    """ T: tiempo para hacer la trayectoria
        alt_ini: Altura inicial
        alt_fin: Alruta final
    """

    # Se hace el cálculo de los coeficientes de la trayectoria
    A = np.matrix([[0, 0, 0, 0, 0, 1], [T**5, T**4, T**3, T**2, T, 1], 
                    [0, 0, 0, 0, 1, 0], [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0], 
                    [0, 0, 0, 2, 0, 0], [20*T**3, 12*T**2, 6*T, 2, 0, 0]])
    b = np.matrix([[alt_ini],[alt_fin],[0],
                    [0], [0], [0]]) 
    return (A**-1)*b

# Hilos para el control y la detección de grietas
def control_vehiculo():
    """En este hilo se tiene el control del vehículo"""

    # Despegue del vehículo
    drone.takeoff()
    print("La altura en el despegue es: ", drone.get_height())
    time.sleep(2)
   
    try:
        # Ganacias del controlador
        PID = {"Kp":1.2, "Kd":0.8, "Ki":0.1} 
        SM = {"rho":0.8, "lambda":0.5, "epsilon":0.5}

        var =  0
        data_error = []
        data_altura = []
        data_time = []
        datos = {"error_alt":[], "alt_des":[], "alt_real":[], 
                 "vel_des":[],"vel_real":[], "tiempo":[], 
                 "dic_estados":[]}

        time_interval = 0.005
        
        # Subir
        for i in range(3):
            # Se toma el valor inicial del tiempo 
            tiempo = 0
            tiempo_ini = time.perf_counter()

            if i==0:
                x = gen_trayectoria(7, 70, 180)
            elif i==2:
                x = gen_trayectoria(7, 180, 70)
            if i==0 or i==2:
                while True:
                    # Cálculo de la altura y velocidad deseada
                    alt_des = (x.item(0)*tiempo**5 + x.item(1)*tiempo**4 
                                + x.item(2)*tiempo**3 + x.item(3)*tiempo**2 
                                + x.item(4)*tiempo + x.item(5))
                    vel_des = (5*x.item(0)*tiempo**4 + 4*x.item(1)*tiempo**3 
                                + 3*x.item(2)*tiempo**2 + 2*x.item(3)*tiempo 
                                + x.item(4))
                    vel_des = 0

                    # Lectura de datos
                    alt_real = drone.get_distance_tof()
                    vel_real = drone.get_speed_z()

                    # Datos adicionales
                    dic_estados = drone.get_current_state()

                    # Cálculo del error
                    error = alt_des - alt_real
                    error_vel = vel_des - vel_real

                    # Superficie de deslizamiento
                    s = error_vel + SM["lambda"] * error
                    
                    # Control PD+SMC de altura
                    u_pid = PID["Kp"]*error + PID["Kd"]*error_vel
                    u_sm = SM["rho"]*np.tanh(s/SM["epsilon"]) + SM["lambda"]*vel_real
                    u = int(np.clip( u_pid + u_sm, -100, 100))
                    drone.send_rc_control(0, 0, u, 0)
                    
                    # Recopilación de datos
                    datos["error_alt"].append(error)
                    datos["alt_des"].append(alt_des)
                    datos["alt_real"].append(alt_real)
                    datos["tiempo"].append(tiempo)
                    datos["vel_des"].append(vel_des)
                    datos["vel_real"].append(vel_real)
                    datos["dic_estados"].append(dic_estados)

                    #data_time.append()
                    time.sleep(time_interval)
                    val_fin = time.perf_counter()
                    tiempo = val_fin - tiempo_ini
                    
                    print("El valor es ", alt_real)
                    if tiempo >= 7:
                        if i==0:
                            print("Acabo la trayectoria 1")
                        elif i==2:
                            print("Acabo la trayectoria 3")
                        break
            if i==1:
                while True:
                    drone.send_rc_control(0, 10, 0, 0)
                    time.sleep(time_interval)
                    val_fin = time.perf_counter()
                    tiempo = val_fin - tiempo_ini
                    
                    #print("El valor es ", tiempo)
                    if tiempo >= 3:
                        print("Acabo en la trayectoria 2")
                        break

        # Escribir valores en un archivo CSV
        df = pd.DataFrame(datos)
        df.to_csv('datos.csv')
        print("Se acabo el vuelo y ya esta el archivo csv")
        drone.land()        
    
    except Exception as e:
        print('Error: ', e)
        drone.land()

    # Se grafican los resultados
    # plt.plot(data_altura)
    # plt.show()

def deteccion_grietas():
    while True:
        print('Grieta detectada')
        time.sleep(1)

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[0]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)

#trayec = threading.Thread(target=trayectoria)
#trayec.start()

if __name__ == '__main__':

    # Se instancia objeto para conectarse con el vehículo
    drone = tello.Tello()
    drone.connect()
    # Inicializa la transmición de video
    drone.streamon()
    # Manejo de hilo de control
    #control_veh = threading.Thread(target=control_vehiculo)
    #control_veh.start()
    
    #det_grietas = threading.Thread(target=deteccion_grietas)
    #det_grietas.start()

    # Se carga el modelo YOLO entrenado
    model = cv2.dnn.readNet("yolov3_custom_train_4000.weights", "yolov3_custom_train.cfg")
    classes = []
    with open("yolo.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = model.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in model.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
	
    """
        cap = start_webcam()
        while True:
            _, frame = cap.read()
            height, width, channels = frame.shape
            blob, outputs = detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()

    """
    # Se toma el valor inicial del tiempo 
    tiempo = 0
    tiempo_ini = time.perf_counter()    
    
    while True:
        #print(alt_des1)
        frame = drone.get_frame_read().frame
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        
        #time.sleep(time_interval)
        val_fin = time.perf_counter()
        tiempo = val_fin - tiempo_ini
        print("El tiempo es:", tiempo)

        if tiempo>30 or (cv2.waitKey(1) & 0xFF == ord('q')):
            print('Finalizo')
            #me.land()
            break
        cv2.destroyAllWindows()
    # Se finaliza la comunicación con el drone
    cv2.destroyAllWindows()
    drone.streamoff()
    drone.end()
    """
    while True:
        frame = drone.get_frame_read().frame
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Finalizo')
            #me.land()
            break
    """

