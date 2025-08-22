# TUTORIAIS

Instalando as dependências da câmera
   Link para instalar as dependencias da câmera (abra em uma nova guia): <a href="https://docs.luxonis.com/software/depthai/manual-install/#Manual%20DepthAI%20installation-Installing%20dependencies">Clique aqui</a>
 


<!-- Próximo tópico -->


# 📚 Tutoriais

Após realizar o download e a instalação das dependências da câmera, assim como a instalação correta da OpenCV, já é possível executar alguns exemplos práticos. Esses exemplos podem ser feitos com a câmera OAK-D ou, caso você não possua a câmera no momento, podem ser adaptados para a webcam do notebook ou PC.
  
Para mais detalhes, esses e mais exemplos podem ser encontrados no site oficial da Luxonis em <a href="https://docs.luxonis.com/">Docs Luxonis</a>.
   
## 👋 Hello World

Esse exemplo foi retirado do site da Luxonis e pode ser executado tanto na câmera OAK-D quanto na câmera do seu notebook/PC.

⚠️ Atenção: antes de rodar o código, certifique-se de selecionar o interpretador Python correto — aquele em que você instalou o OpenCV, o DepthAI e as demais dependências. Recomenda-se que essas bibliotecas sejam instaladas e configuradas dentro de um ambiente virtual (venv) para garantir isolamento e evitar conflitos com outros projetos.
    
Vamos mergulhar nos conceitos básicos usando um exemplo. Vamos criar uma aplicação simples que executa uma rede neural de detecção de objetos e transmite vídeo em cores com        as detecções da rede neural visualizadas. Usaremos a API Python do DepthAI para criar a aplicação.

O primeiro nó que adicionaremos é o **ColorCamera**. Esse nó selecionará automaticamente a câmera central (que, na maioria dos dispositivos, é a câmera de cor) e fornecerá o fluxo de vídeo para o próximo nó no pipeline.
Usaremos a saída **preview**, redimensionada para 300x300, de forma a se ajustar ao tamanho de entrada do **mobilenet-ssd** (que definiremos mais adiante).
    
### Câmera

``
# first, import all necessary modules
from pathlib import Path

import blobconverter
import cv2
import depthai
import numpy as np


pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

detection_nn = pipeline.createMobileNetDetectionNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
# Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
detection_nn.setConfidenceThreshold(0.5)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

cam_rgb.preview.link(xout_rgb.input)
cam_rgb.preview.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None
    detections = []

    # Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
    # receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    while True:
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            # when data from nn is received, we take the detections array that contains mobilenet-ssd results
            detections = in_nn.detections


        if frame is not None:
            for detection in detections:
                # for each bounding box, we first normalize it to match the frame size
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("preview", frame)

        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break
```







### Notebook


   import cv2
   import numpy as np
   
   # ----------------- CONFIGURAÇÃO -----------------
   # Confiança mínima para exibir detecções
   CONFIDENCE_THRESHOLD = 0.5
   
   # Caminhos dos arquivos do modelo MobileNet-SSD
   # Você precisa baixar esses arquivos:
   # 1. prototxt: https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt
   # 2. caffemodel: https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.caffemodel
   MODEL_PROTOTXT = "mobilenet_ssd.prototxt"
   MODEL_WEIGHTS = "mobilenet_ssd.caffemodel"
   
   # ----------------- INICIALIZAÇÃO -----------------
   cap = cv2.VideoCapture(0)  # 0 = webcam interna
   if not cap.isOpened():
       raise RuntimeError("Não foi possível abrir a webcam do notebook")
   
   # Carrega o modelo MobileNet-SSD
   net = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_WEIGHTS)
   
   # Lista de classes que o MobileNet-SSD detecta
   CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]
   
   # ----------------- LOOP PRINCIPAL -----------------
   while True:
       ret, frame = cap.read()
       if not ret:
           break
   
       # Redimensiona e pré-processa o frame
       blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
       net.setInput(blob)
       detections = net.forward()
   
       # Percorre todas as detecções
       for i in range(detections.shape[2]):
           confidence = detections[0, 0, i, 2]
           if confidence > CONFIDENCE_THRESHOLD:
               idx = int(detections[0, 0, i, 1])
               box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                                          frame.shape[1], frame.shape[0]])
               (startX, startY, endX, endY) = box.astype("int")
   
               # Desenha o retângulo e a classe detectada
               label = f"{CLASSES[idx]}: {confidence*100:.1f}%"
               cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
               y = startY - 15 if startY - 15 > 15 else startY + 15
               cv2.putText(frame, label, (startX, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
   
       # Mostra o frame com as detecções
       cv2.imshow("Webcam Preview", frame)
   
       # Sai ao pressionar 'q'
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
   cap.release()
   cv2.destroyAllWindows()
