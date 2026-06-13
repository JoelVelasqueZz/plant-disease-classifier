# Clasificación de Enfermedades en Plantas con Deep Learning

Proyecto de clasificación de imágenes usando **PyTorch** y **Transfer Learning** (ResNet18), 
para identificar enfermedades en hojas de cultivos de tomate, papa y pimiento.

## Resultados

- **Precisión en validación: 98.33%**
- Dataset: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (20,638 imágenes, 15 clases)
- Arquitectura: ResNet18 preentrenado (ImageNet), fine-tuning de `layer4` + capa final
- Data augmentation: rotación, flip horizontal, brillo/contraste
- 10 épocas de entrenamiento, ~62s por época (GPU T4)

## Demo interactiva

Probá el modelo en vivo (subí una foto o usá la cámara):
[plant-disease-demo](https://huggingface.co/spaces/JoelVela/plant-disease-demo)

![Resultados de entrenamiento](grafico_entrenamiento.png)

## Matriz de Confusión

![Matriz de confusión](matriz_confusion.png)

## Predicciones de ejemplo

![Predicciones](predicciones_ejemplo.png)

## Dataset

15 clases de hojas (tomate, papa, pimiento), sanas y con distintas enfermedades:
- Bacterial spot, Early/Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus

## Tecnologías

- Python, PyTorch, torchvision
- Transfer Learning con ResNet18 (fine-tuning)
- Data Augmentation
- Google Colab (GPU T4)
- scikit-learn, matplotlib, seaborn

## Modelo entrenado

El modelo entrenado (`.pth`) está disponible en Hugging Face: 
[JoelVela/plant-disease-resnet18](https://huggingface.co/JoelVela/plant-disease-resnet18)

## Limitaciones y prueba con imágenes externas

El dataset PlantVillage contiene imágenes con fondo uniforme y condiciones 
controladas. Para evaluar generalización real, se probó el modelo con 
3 imágenes externas (fondos de jardín/campo real, no presentes en el dataset):

![Prueba 1](prueba1.png)
![Prueba 2](prueba2.png)
![Prueba 3](prueba3.png)

**Observaciones:**
- En el caso visualmente más claro (Prueba 3, manchas grandes irregulares, 
  patrón clásico de Late Blight), el modelo predijo correctamente con 99.3% 
  de confianza.
- En casos más ambiguos o con fondos complejos, la confianza bajó 
  notablemente (Prueba 2: 62.1%), y la Prueba 1 (97.8%) podría corresponder 
  a una confusión con Septoria leaf spot, enfermedad visualmente similar.

**Conclusión:** el alto accuracy en validación (98.33%) refleja el 
desempeño bajo condiciones similares al dataset de entrenamiento. La 
confianza del modelo varía según la claridad del patrón y la similitud 
con las condiciones de PlantVillage, lo cual es un punto de mejora para 
trabajo futuro (ej. incluir imágenes de campo real como PlantDoc).

## Conclusiones

- El modelo alcanzó **98.33% de precisión** combinando fine-tuning de 
  `layer4` con data augmentation, una mejora significativa frente al 
  88.44% inicial (solo última capa entrenable, sin augmentation).
- Las pruebas con imágenes externas revelaron buen desempeño en casos 
  visualmente claros, pero menor confianza en casos ambiguos o con 
  fondos complejos.
- Posibles mejoras futuras: incluir imágenes de campo real (dataset 
  PlantDoc), probar ResNet50, y desarrollar una demo interactiva con 
  Gradio en Hugging Face Spaces.

### Aplicación práctica
Este tipo de modelo podría integrarse en una aplicación móvil que permita
a pequeños agricultores tomar una foto de una hoja y recibir un diagnóstico
preliminar, facilitando la detección temprana de enfermedades en cultivos. 
Para uso en campo real sería necesario entrenar con imágenes en condiciones 
similares a las del usuario final.

## Autor

**Joel Velasquez** — Estudiante de Ingeniería en TI, UTMACH  
[LinkedIn](https://linkedin.com/in/joel-velasquez-827923278) | [GitHub](https://github.com/JoelVelasqueZz)
