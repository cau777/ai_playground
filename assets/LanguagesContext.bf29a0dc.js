import{c as s,b as n,u as r}from"./entry-client.bdc462ff.js";const i={digits:{title:"Digits Recognition",probabilityLabel:"Probability",evaluateBtn:"Evaluate",instructions:"Draw a single digit from 0 to 9. The program uses AI to try guessing it and shows the results in the graph below, according to the probabilities of each number. In case the result is wrong, try rewriting the digit a bit different.",limitations:"The model has a tested accuracy of 90% and usually fails with 7 and 9.",examples:"Examples:"},chess:{title:"Chess",opening:"Opening",loading:"Loading...",start:"Start game",gameResult:"Game Result",gameResultUndefined:"Undefined",gameResultAborted:"Aborted",gameResultFiftyMoveRule:"Draw by the Fifty Move Rule",gameResultRepetition:"Draw by 3-fold repetition",gameResultStalemate:"Stalemate",gameResultInsufficientMaterial:"Draw by insufficient material",gameResultCheckmateWhite:"White won by checkmate",gameResultCheckmateBlack:"Black won by checkmate",playerTurn:"It's your turn to move",aiTurn:"The AI is deciding...",playAs:"Play as:",openingsBook:"Openings book:",none:"None",complete:"Complete",gambits:"Gambits",whiteSide:"White",blackSide:"Black",randomSide:"Random",mainlines:"Main lines"},home:{title:"Home",welcome:"Welcome",description:"This is a big project that I'm developing in my free time. It aims to explore some of the capabilities of AI in many interactive projects. To achieve that, I'm writing a Deep Learning library in Rust almost from scratch. All the code is open-source in my ",statusEarlyDev:"Early development"}},l={digits:{title:"Reconhecimento de D\xEDgitos",probabilityLabel:"Probabilidade",evaluateBtn:"Avaliar",instructions:"Desenhe um \xFAnico d\xEDgito de 0 a 9. O programa usa t\xE9cnicas de IA para tentar adivinhar e mostra os resultados no gr\xE1fico, conforme a probabilidade de cada d\xEDgito. Caso, o resultado n\xE3o fa\xE7a sentido, tente reescrever o n\xFAmero um pouco diferente.",limitations:"O modelo tem uma precis\xE3o testada de 90% e erra principalmente 7 e 9.",examples:"Exemplos:"},chess:{title:"Xadrez",opening:"Abertura",loading:"Carregando...",start:"Novo jogo",gameResult:"Conclus\xE3o do jogo",gameResultUndefined:"Indefinido",gameResultAborted:"Abortado",gameResultFiftyMoveRule:"Empate pela Regra dos Cinquenta Movimentos",gameResultRepetition:"Empate por repeti\xE7\xE3o de 3 n\xEDveis",gameResultStalemate:"Empate por afogamento",gameResultInsufficientMaterial:"Empate por material insuficiente",gameResultCheckmateWhite:"Brancas ganharam por cheque-mate",gameResultCheckmateBlack:"Pretas ganharam por cheque-mate",playerTurn:"\xC9 sua vez de jogar",aiTurn:"A IA est\xE1 decidindo...",playAs:"Jogar como:",openingsBook:"Repert\xF3rio de abertura:",none:"Nenhum",complete:"Completo",gambits:"G\xE2mbitos",whiteSide:"Brancas",blackSide:"Pretas",randomSide:"Aleat\xF3rio",mainlines:"Principais"},home:{title:"P\xE1gina Inicial",welcome:"Bem-vindo(a)",description:"Esse \xE9 um grande projeto que estou desenvolvendo em meu tempo livre. Ele visa explorar algumas capacidades de IA em v\xE1rios projetos interativos. Para isso, estou escrevendo uma biblioteca de Deep Learning em Rust, praticamente do zero. Todo o c\xF3digo se encontra open-source no meu ",statusEarlyDev:"Fase inicial"}},o=n();function m(a){let e;switch(a){case"pt":e=l;break;case"en":e=i;break;default:e=i;break}return{lang:a,t:e}}const u=a=>{let e=m(a.lang);return s(o.Provider,{value:e,get children(){return a.children}})},c=()=>r(o),t=()=>c().t,g=()=>t().digits,p=()=>t().chess,h=()=>t().home;export{u as L,g as a,p as b,h as c,c as u};