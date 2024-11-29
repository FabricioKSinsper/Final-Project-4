## WGSL Raymarcher Template
Template básico de um Ray Marcher em WGSL (WEBGPU). Tudo o que precisa para fazer o projeto esta aqui.

### ⚠️ Atenção

O projeto provavelmente não vai rodar em linux, já que a GPU não fica acessível pro WGSL.

### Instruções
Faça um fork do projeto, clone e abra no Visual Studio (ou outro software, mas recomendo esse)

Baixe (se no visual studio) -> **Name: Live Server - VS Marketplace Link:** [link para o marketplace](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer)

Aperte ```CTRL + Shift + P``` e selecione ```Live Server```. Um browser com o projeto rodando deve abrir.

### Funções para implementar/complementar (olhe a sessão de notas para mais detalhes)
- ```sdf_round_box```
- ```sdf_sphere```
- ```sdf_torus```
- ```get_light```
- ```get_soft_shadow```
- ```get_normal```
- ```march```
- ```repeat```
- ```scene```
- ```transform_p```
- ```op_smooth_union```
- ```op_smooth_subtraction```
- ```op_smooth_intersection```
- ```preprocess```
- ```animate```
- ```render```

### Controles
Você pode controlar qualquer parametro com o GUI ao lado.

**Dica**: Você pode clicar no parametro e fazer um scroll para mudar ele de maneira smooth.
Para mover a camera, use WASDQEXZ. Para rodar, use as setas.

### Dicas
Procure pelos comentários no código para te ajudar. Dê uma olhada nos outros arquivos ```.wgsl``` além do ```raymarcher.wgsl```, várias funções que você vai precisar já estão disponíveis lá, prontas para usar.

Depois que criar uma cena nova, basta apertar ```PrintScene``` (copia para o clipboard automatico) e colar no arquivo ```scenes.js```.
O reconhecimento é automático. 

### Nota
Você pode calcular sua nota com base nas cenas que você fez. Você sempre pode comparar aqui (https://gubebra.itch.io/raymarching-webgpu)
- C: ```"Sphere", "SkyAndHS", "Multiple"```
- C+: ```"Rotation"```
- B: ```"Animation"```, ```"Outline"```
- B+: ```"Union"```, ```"Subtraction"```, ```"Intersection"```, ```"Blobs"```
- A: ```"Mod"```, ```"SoftShadows"```
- A+: Adicione uma nova primitiva geometrica ou ```"Fractal" ou "Weird"```. Você deve fazer sua própria função de fractal ou uma cena com uma geometria maluca.

+ Meio conceito: veja as dicas e creie uma cena nova interessante (não somente uma esfera e pronto)

### Entrega:
Via Blackboard, entregue o link do git.

#### Dia 27/11/24
