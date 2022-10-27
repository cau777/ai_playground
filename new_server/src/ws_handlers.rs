use crate::{TaskManagerDep, ClientsDep};
use crate::{utils::EndpointResult, CurrentModelDep, ModelsSourcesDep};
use codebase::integration::compression::{compress_default, decompress_default};
use codebase::integration::proto_loading::load_model_from_bytes;
use futures::SinkExt;
use futures::{
    channel::mpsc,
    FutureExt, StreamExt,
};
use warp::ws::Message;
use warp::{
    ws::{WebSocket, Ws},
    Reply,
};

struct Deps {
    current_model: CurrentModelDep,
    sources: ModelsSourcesDep,
    task_manager: TaskManagerDep,
    clients: ClientsDep,
}

pub async fn ws_train_handler(
    ws: Ws,
    current_model: CurrentModelDep,
    sources: ModelsSourcesDep,
    task_manager: TaskManagerDep,
    clients: ClientsDep,
) -> EndpointResult<impl Reply> {
    let result = ws.on_upgrade(move |socket| {
        let deps = Deps {
            current_model: current_model.clone(),
            sources: sources.clone(),
            task_manager: task_manager.clone(),
            clients: clients.clone(),
        };
        
        client_connection(socket, deps)
    });
    Ok(result)
}

async fn client_connection(socket: WebSocket, deps: Deps) {
    let (client_ws_sender, mut client_ws_rcv) = socket.split();
    let (client_sender, client_rcv) = mpsc::unbounded();
    
    {
        deps.clients.write().await.register(client_sender);
    }

    tokio::task::spawn(client_rcv.forward(client_ws_sender).map(|result| {
        if let Err(e) = result {
            eprintln!("Error sending websocket msg: {}", e);
        }
    }));

    while let Some(result) = client_ws_rcv.next().await {
        let msg = match result {
            Ok(msg) => msg,
            Err(e) => {
                eprintln!("error receiving ws message): {}", e);
                break;
            }
        };
        
        // TODO: return error feedback
        let err = handle_message_receive(msg, &deps).await;
        if let Err(e) = err {
            eprintln!("{}", e);
        }
    }
}

async fn handle_message_receive(msg: Message, deps: &Deps) -> Result<(), String> {
    println!("Start handle receive");
    let bytes = msg.as_bytes();
    let bytes = decompress_default(bytes).map_err(|e| format!("Error decompressing: {}", e))?;
    let deltas = load_model_from_bytes(&bytes).ok_or("Error parsing proto")?;
    let Deps {current_model, sources, task_manager, clients} = deps;

    let mut current_model = current_model.write().await;
    current_model.increment(1, deltas);
    if current_model.should_save() {
        let mut sources = sources.write().await;
        current_model.save_to(&mut sources).unwrap();
        
        task_manager.write().await.add_to_test(current_model.version(), sources.test_count());
    }

    println!("Finish handle receive");
    Ok(())
}
