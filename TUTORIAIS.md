# TUTORIAIS

Instalando as dependências da câmera
   Link para instalar as dependencias da câmera (abra em uma nova guia): <a href="https://docs.luxonis.com/software/depthai/manual-install/#Manual%20DepthAI%20installation-Installing%20dependencies">Clique aqui</a>
 


<!-- Próximo tópico -->


# 📚 Tutoriais

Após realizar o download e a instalação das dependências da câmera, assim como a instalação correta da OpenCV, já é possível executar alguns exemplos práticos. Esses exemplos podem ser feitos com a câmera OAK-D ou, caso você não possua a câmera no momento, podem ser adaptados para a webcam do notebook ou PC.
  
Para mais detalhes, esses e mais exemplos podem ser encontrados no site oficial da Luxonis em <a href="https://docs.luxonis.com/">Docs Luxonis</a>.

<details>
summary>## 👋 Hello World</summary>

Esse exemplo foi retirado do site da Luxonis e pode ser executado tanto na câmera OAK-D quanto na câmera do seu notebook/PC.

⚠️ Atenção: antes de rodar o código, certifique-se de selecionar o interpretador Python correto — aquele em que você instalou o OpenCV, o DepthAI e as demais dependências. Recomenda-se que essas bibliotecas sejam instaladas e configuradas dentro de um ambiente virtual (venv) para garantir isolamento e evitar conflitos com outros projetos.
    
Vamos mergulhar nos conceitos básicos usando um exemplo. Vamos criar uma aplicação simples que executa uma rede neural de detecção de objetos e transmite vídeo em cores com        as detecções da rede neural visualizadas. Usaremos a API Python do DepthAI para criar a aplicação.

O primeiro nó que adicionaremos é o **ColorCamera**. Esse nó selecionará automaticamente a câmera central (que, na maioria dos dispositivos, é a câmera de cor) e fornecerá o fluxo de vídeo para o próximo nó no pipeline.
Usaremos a saída **preview**, redimensionada para 300x300, de forma a se ajustar ao tamanho de entrada do **mobilenet-ssd** (que definiremos mais adiante).
    
### Câmera

```
# Primeiro, importamos todos os módulos necessários
from pathlib import Path

import blobconverter
import cv2
import depthai
import numpy as np


pipeline = depthai.Pipeline()

# Primeiro, queremos a câmera de cor como saída
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)  # 300x300 será o tamanho do frame de pré-visualização, disponível como saída 'preview' do nó
cam_rgb.setInterleaved(False)

detection_nn = pipeline.createMobileNetDetectionNetwork()

# O blob é o arquivo da Rede Neural, compilado para MyriadX. Ele contém tanto a definição quanto os pesos do modelo
# Estamos usando a ferramenta blobconverter para obter automaticamente o blob do MobileNetSSD a partir do OpenVINO Model Zoo

detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))

# Em seguida, filtramos as detecções que estão abaixo de um limite de confiança. A confiança pode estar entre <0..1>
detection_nn.setConfidenceThreshold(0.5)

# XLinkOut é uma "saída" do dispositivo. Qualquer dado que você queira transferir para o host precisa ser enviado via XLink
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

cam_rgb.preview.link(xout_rgb.input)
cam_rgb.preview.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)

# O pipeline agora está finalizado e precisamos encontrar um dispositivo disponível para executá-lo
# Estamos usando um context manager aqui, que irá liberar o dispositivo após o uso
with depthai.Device(pipeline) as device:
    # A partir deste ponto, o dispositivo estará em modo "executando" e começará a enviar dados via XLink

    # Para consumir os resultados do dispositivo, obtemos duas filas de saída com os nomes de stream definidos anteriormente
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    # Aqui alguns valores padrão são definidos. Frame será uma imagem do stream "rgb" e detections conterá os resultados da rede neural
    frame = None
    detections = []

    # Como as detecções retornadas pela rede neural possuem valores no intervalo <0..1>, 
    # eles precisam ser multiplicados pela largura/altura do frame para obter a posição real da caixa delimitadora na imagem
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    while True:
        # Tentamos buscar os dados das filas nn/rgb. tryGet retorna ou o pacote de dados ou None se não houver nada
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            # Se o pacote da câmera RGB estiver presente, recuperamos o frame no formato OpenCV usando getCvFrame
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            # Quando os dados da rede neural são recebidos, pegamos o array de detecções que contém os resultados do mobilenet-ssd
            detections = in_nn.detections


        if frame is not None:
            for detection in detections:
                # Para cada caixa delimitadora, primeiro normalizamos para corresponder ao tamanho do frame
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                # E então desenhamos um retângulo no frame para mostrar o resultado
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # Após todo o desenho estar concluído, mostramos o frame na tela
            cv2.imshow("preview", frame)

        # A qualquer momento, você pode pressionar "q" para sair do loop principal, encerrando o programa
        if cv2.waitKey(1) == ord('q'):
            break

```







### Notebook

```
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
```

## Para fechar as janelas basta apertar "q"
</details>
