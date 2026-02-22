# Identificação do Projeto de Inteligência Artificial
## Nome dos Alunos responsáveis:
- Enzo Emanuel Maia Costa 
- Jackson Santana Carvalho Junior
- Adam Guilherme Mendes Lima
- Matheus Henrique Silva de Melo
- Samuel Guimarães Silva
 





## Resumo descritivo

A área de Inteligência Artificial tem avançado significativamente com a popularização dos Grandes Modelos de Linguagem (LLMs), sistemas capazes de gerar e processar linguagem natural de forma altamente coerente, e dentro desse contexto, a orquestração de chamadas e a gestão de contexto destacam-se por permitir que um agente converse de forma fluida, "lembrando-se" do que foi dito anteriormente. Este trabalho tem como objetivo a implementação de um Agente Conversacional (Chatbot com memória), aplicando os conceitos de consumo de APIs e estruturação de pipelines a um problema prático e essencial: a manutenção do histórico de conversa, onde a escolha desse problema visa facilitar a compreensão dos desafios lógicos de uma aplicação baseada em LLM sem o uso de banco de dados, bem como permitir uma clara correspondência entre o pseudocódigo apresentado em aula e a implementação prática desenvolvida pela equipe. O foco principal do projeto é evidenciar como a manipulação do histórico e o truncamento de mensagens evitam o estouro de tokens do modelo, garantindo requisições eficientes, dessa forma, o projeto prioriza a clareza conceitual, a fidelidade ao pseudocódigo e a facilidade de compreensão da orquestração.





## Descrição do Problema
O problema abordado neste trabalho consiste em permitir que um sistema interativo atue como um assistente virtual capaz de manter uma conversa contínua com o usuário, gerenciando dinamicamente o histórico de interações, o ambiente do sistema é estruturado em torno da comunicação com uma API externa, onde cada interação exige que as mensagens anteriores sejam reenviadas para fornecer contexto, e o grande desafio lógico reside no fato de que essas APIs possuem limites rígidos de entrada (tokens), caso o histórico cresça indefinidamente, a requisição falhará. O problema caracteriza-se, portanto, como um cenário de Engenharia de Prompt e Orquestração , no qual o algoritmo deve aplicar regras de truncamento — como a remoção das mensagens mais antigas (excluindo instruções de sistema)  — garantindo que o volume de dados se mantenha dentro da capacidade de processamento do modelo, maximizando a coerência e utilidade das respostas ao longo da conversa.


## Visualização do Projeto

Para facilitar a visualização do aprendizado do agente, do ambiente estocástico e garantir a interatividade do projeto, 
implementamos uma interface gráfica completa utilizando a biblioteca Streamlit. O projeto também foi implantado na nuvem e
pode ser testado diretamente pelo navegador através do **Hugging Face Spaces**: 

🔗 **[Clique aqui para acessar a Simulação Online](https://huggingface.co/spaces/TastelessNeutrino/LLM-Grupo11)**
